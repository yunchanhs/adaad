import time
import pyupbit
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# 설정값
STOP_LOSS_THRESHOLD = -0.03  # -3% 손절
TAKE_PROFIT_THRESHOLD = 0.05  # +5% 익절
COOLDOWN_TIME = timedelta(minutes=5)
ML_THRESHOLD = 0.02  # 매수 신호 임계값
ML_SELL_THRESHOLD = -0.02  # 매도 신호 임계값

recent_trades = {}
entry_prices = {}
highest_prices = {}  # 최고가 기록용

# 모델 학습 주기 관련 변수
last_trained_time = None  # 마지막 학습 시간
TRAINING_INTERVAL = timedelta(hours=6)  # 6시간마다 재학습

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_top_tickers(n=10):
    """거래량 상위 n개 코인을 선택"""
    tickers = pyupbit.get_tickers(fiat="KRW")
    volumes = []
    for ticker in tickers:
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
            volumes.append((ticker, df['volume'].iloc[-1]))
        except:
            volumes.append((ticker, 0))
    sorted_tickers = sorted(volumes, key=lambda x: x[1], reverse=True)
    return [ticker for ticker, _ in sorted_tickers[:n]]

def detect_surge_tickers(threshold=0.03):
    """실시간 급상승 코인을 감지"""
    tickers = pyupbit.get_tickers(fiat="KRW")
    surge_tickers = []
    for ticker in tickers:
        try:
            df = pyupbit.get_ohlcv(ticker, interval="minute1", count=5)
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            if price_change >= threshold:
                surge_tickers.append(ticker)
        except:
            continue
    return surge_tickers

# 머신러닝 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model, num_heads, num_layers, num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = x[:, -1, :]  # 마지막 시퀀스의 출력
        return self.fc_out(x)

# 지표 계산 함수 (생략, 기존 코드 동일)
# get_macd, get_rsi, get_adx, get_atr, get_features

def get_macd(ticker):
    """주어진 코인의 MACD와 Signal 라인을 계산하는 함수"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)  # 5분봉 데이터 가져오기
    df['short_ema'] = df['close'].ewm(span=12, adjust=False).mean()  # 12-period EMA
    df['long_ema'] = df['close'].ewm(span=26, adjust=False).mean()   # 26-period EMA
    df['macd'] = df['short_ema'] - df['long_ema']  # MACD = Short EMA - Long EMA
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()  # Signal line = 9-period EMA of MACD
    return df['macd'].iloc[-1], df['signal'].iloc[-1]  # 최신 값 반환

def get_rsi(ticker, period=14):
    """주어진 코인의 RSI (Relative Strength Index)를 계산하는 함수"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)  # 5분봉 데이터 가져오기
    delta = df['close'].diff()  # 종가 차이

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # 상승분의 평균
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # 하락분의 평균

    rs = gain / loss  # 상대 강도
    rsi = 100 - (100 / (1 + rs))  # RSI 계산

    return rsi.iloc[-1]  # 최신 RSI 값 반환

def get_adx(ticker, period=14):
    """주어진 코인의 ADX (Average Directional Index)를 계산하는 함수"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)  # 5분봉 데이터 가져오기

    # True Range 계산
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = abs(df['high'] - df['close'].shift(1))
    df['L-C'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)  # True Range

    # +DM, -DM 계산
    df['+DM'] = df['high'] - df['high'].shift(1)
    df['-DM'] = df['low'].shift(1) - df['low']
    df['+DM'] = df['+DM'].where(df['+DM'] > df['-DM'], 0)
    df['-DM'] = df['-DM'].where(df['-DM'] > df['+DM'], 0)

    # Smoothed TR, +DM, -DM
    df['TR_smooth'] = df['TR'].rolling(window=period).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=period).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period).sum()

    # +DI, -DI 계산
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

    # ADX 계산
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(window=period).mean()  # ADX

    return df['ADX'].iloc[-1]  # 최신 ADX 값 반환

def get_atr(ticker, period=14):
    """주어진 코인의 ATR (Average True Range)을 계산하는 함수"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)  # 5분봉 데이터 가져오기

    # True Range 계산
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = abs(df['high'] - df['close'].shift(1))
    df['L-C'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)  # True Range

    # ATR 계산
    df['ATR'] = df['TR'].rolling(window=period).mean()

    return df['ATR'].iloc[-1]  # 최신 ATR 값 반환

def get_features(ticker):
    """코인의 과거 데이터와 지표를 가져와 머신러닝에 적합한 피처 생성"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=1000)

    # MACD 및 Signal 계산
    df['macd'], df['signal'] = get_macd(ticker)  # get_macd 함수 호출

    # RSI 계산
    df['rsi'] = get_rsi(ticker)  # get_rsi 함수 호출

    # ADX 계산
    df['adx'] = get_adx(ticker)  # get_adx 함수 호출

    # ATR 계산
    df['atr'] = get_atr(ticker)  # get_atr 함수 호출

    df['return'] = df['close'].pct_change()  # 수익률
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1  # 미래 수익률

    # NaN 값 제거
    df.dropna(inplace=True)
    return df

# 거래 관련 함수 (생략, 기존 코드 동일)
# get_balance, buy_crypto_currency, sell_crypto_currency

def get_balance(ticker):
    """잔고 확인"""
    try:
        upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
        balance = upbit.get_balance(ticker)
        return balance
    except Exception as e:
        print(f"[{ticker}] 잔고 확인 중 에러 발생: {e}")
        return 0

def buy_crypto_currency(ticker, amount):
    """시장가로 코인 매수"""
    try:
        upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
        order = upbit.buy_market_order(ticker, amount)
        return order
    except Exception as e:
        print(f"[{ticker}] 매수 중 에러 발생: {e}")
        return None

def sell_crypto_currency(ticker, amount):
    """시장가로 코인 매도"""
    try:
        upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
        order = upbit.sell_market_order(ticker, amount)
        return order
    except Exception as e:
        print(f"[{ticker}] 매도 중 에러 발생: {e}")
        return None

# 데이터셋 및 모델 학습 함수
class TradingDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx+self.seq_len][['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].values
        y = self.data.iloc[idx + self.seq_len]['future_return']
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_transformer_model(ticker, epochs=10):  # 에포크 기본값 10으로 설정
    print(f"모델 학습 시작: {ticker}")
    data = get_features(ticker)
    seq_len = 30
    dataset = TradingDataset(data, seq_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # 배치 크기 32로 설정

    input_dim = 6
    d_model = 64
    num_heads = 8
    num_layers = 2
    output_dim = 1

    model = TransformerModel(input_dim, d_model, num_heads, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    print(f"모델 학습 완료: {ticker}")
    return model

def get_ml_signal(ticker, model):
    """AI 신호 계산"""
    try:
        features = get_features(ticker)
        latest_data = features[['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].tail(30)
        X_latest = torch.tensor(latest_data.values, dtype=torch.float32).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            prediction = model(X_latest).item()
        return prediction
    except Exception as e:
        print(f"[{ticker}] AI 신호 계산 에러: {e}")
        return 0

def should_retrain_model():
    """6시간마다 모델을 재학습하도록 결정하는 함수"""
    global last_trained_time
    if last_trained_time is None:
        return True
    return datetime.now() - last_trained_time >= TRAINING_INTERVAL

# 모델 학습 시작
def retrain_models_if_needed(tickers):
    """모델이 6시간 주기로 재학습 필요 시 재학습"""
    global last_trained_time
    if should_retrain_model():
        print("모델을 재학습합니다...")
        for ticker in tickers:
            print(f"모델 학습 시작: {ticker}")
            models[ticker] = train_transformer_model(ticker, epochs=50)  # 50 에포크로 학습
        last_trained_time = datetime.now()
        print(f"모델 학습 완료: {last_trained_time}")

# 메인 로직
if __name__ == "__main__":
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    print("자동매매 시작!")

    tickers = pyupbit.get_tickers(fiat="KRW")
    models = {}

    # 초기 설정
    top_tickers = get_top_tickers(n=10)
    print(f"거래량 상위 코인: {top_tickers}")
    models = {ticker: train_transformer_model(ticker) for ticker in top_tickers}
    recent_surge_tickers = {}

    try:
        while True:
            # 1. 상위 코인 업데이트 (6시간마다)
            if datetime.now().hour % 6 == 0 and datetime.now().minute == 0:
                top_tickers = get_top_tickers(n=10)
                print(f"[{datetime.now()}] 상위 코인 업데이트: {top_tickers}")

                # 새롭게 추가된 코인 모델 학습
                for ticker in top_tickers:
                    if ticker not in models:
                        models[ticker] = train_transformer_model(ticker)

            # 2. 급상승 코인 감지
            surge_tickers = detect_surge_tickers(threshold=0.03)
            for ticker in surge_tickers:
                if ticker not in recent_surge_tickers:
                    print(f"[{datetime.now()}] 급상승 감지: {ticker}")
                    recent_surge_tickers[ticker] = datetime.now()
                    if ticker not in models:
                        models[ticker] = train_transformer_model(ticker, epochs=10)  # 급상승 코인은 빠르게 학습

                # 쿨다운 타임 체크
                if ticker in recent_trades and datetime.now() - recent_trades[ticker] < COOLDOWN_TIME:
                    continue

                try:
                    # AI 및 지표 계산
                    ml_signal = get_ml_signal(ticker, models[ticker])
                    macd, signal = get_macd(ticker)
                    rsi = get_rsi(ticker)
                    adx = get_adx(ticker)
                    current_price = pyupbit.get_current_price(ticker)

                    # 매수 조건
                    if ml_signal > ML_THRESHOLD and macd > signal and rsi < 30 and adx > 25 and krw_balance > 5000:
                        buy_amount = krw_balance * 0.3
                        buy_result = buy_crypto_currency(ticker, buy_amount)
                        if buy_result:
                            entry_prices[ticker] = current_price
                            highest_prices[ticker] = current_price
                            recent_trades[ticker] = datetime.now()
                            print(f"[{ticker}] 매수 완료: {buy_amount:.2f}원, 가격: {current_price:.2f}")

                    # 매도 조건
                    elif ticker in entry_prices:
                        entry_price = entry_prices[ticker]
                        highest_prices[ticker] = max(highest_prices[ticker], current_price)
                        change_ratio = (current_price - entry_price) / entry_price

                        # 손절 조건 보완
                        if change_ratio <= STOP_LOSS_THRESHOLD:
                            if ml_signal > ML_THRESHOLD:
                                print(f"[{ticker}] 손실 상태지만 AI 신호 긍정적, 매도 보류.")
                            else:
                                coin_balance = get_balance(ticker.split('-')[1])
                                sell_crypto_currency(ticker, coin_balance)
                                del entry_prices[ticker]
                                del highest_prices[ticker]
                                print(f"[{ticker}] 손절 매도 완료.")

                        # 익절 또는 최고점 하락
                        elif change_ratio >= TAKE_PROFIT_THRESHOLD or current_price < highest_prices[ticker] * 0.98:
                            if ml_signal < ML_SELL_THRESHOLD:
                                coin_balance = get_balance(ticker.split('-')[1])
                                sell_crypto_currency(ticker, coin_balance)
                                del entry_prices[ticker]
                                del highest_prices[ticker]
                                print(f"[{ticker}] 매도 완료 (익절 또는 최고점 하락).")
                            else:
                                print(f"[{ticker}] AI 신호 긍정적, 매도 보류.")

                except Exception as e:
                    print(f"[{ticker}] 처리 중 에러 발생: {e}")

    except KeyboardInterrupt:
        print("프로그램이 종료되었습니다.")
