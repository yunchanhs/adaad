import time
import pyupbit
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader  # DataLoader 클래스를 임포트
import torch.optim as optim
from datetime import datetime  # datetime 모듈에서 datetime 클래스를 임포트
from sklearn.preprocessing import StandardScaler

now = datetime.now()  # now() 메소드 호출

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# 모델 학습 주기 관련 변수
last_trained_time = None  # 마지막 학습 시간
TRAINING_INTERVAL = timedelta(hours=8)  # 6시간마다 재학습

# 매매 전략 관련 임계값
ML_THRESHOLD = 0.5  # 기본값
ML_SELL_THRESHOLD = 0.3  # AI 신호 매도 기준
STOP_LOSS_THRESHOLD = -0.05  # 손절 (-5%)
TAKE_PROFIT_THRESHOLD = 0.1  # 익절 (10%)
COOLDOWN_TIME = timedelta(minutes=5)  # 동일 코인 재거래 쿨다운 시간
SURGE_COOLDOWN_TIME = timedelta(minutes=10) # 급등 코인 쿨다운 시간

# 계좌 정보 저장
entry_prices = {}  # 매수한 가격 저장
highest_prices = {}  # 매수 후 최고 가격 저장
recent_trades = {}  # 최근 거래 기록
recent_surge_tickers = {}  # 최근 급상승 감지 코인 저장

def get_top_tickers(n=5):
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

def get_ohlcv_cached(ticker, interval="minute60"):
    time.sleep(0.2)  # 요청 간격 조절
    return pyupbit.get_ohlcv(ticker, interval=interval)
    
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
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=5000)

    # MACD 및 Signal 계산
    df['macd'], df['signal'] = get_macd(ticker)  # get_macd 함수 호출

    # RSI 계산
    df['rsi'] = get_rsi(ticker)  # get_rsi 함수 호출

    # ADX 계산
    df['adx'] = get_adx(ticker)  # get_adx 함수 호출

    # ATR 계산
    df['atr'] = get_atr(ticker)  # get_atr 함수 호출

    df['return'] = df['close'].pct_change()  # 수익률
    df['future_return'] = df['close'].shift(-3) / df['close'] - 1  # 3개 후 캔들 예측

    # NaN 값 제거
    df.dropna(inplace=True)
    return df

# 거래 관련 함수 (생략, 기존 코드 동일)
# get_balance, buy_crypto_currency, sell_crypto_currency

# Upbit 객체 전역 선언 (한 번만 생성)
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

def get_balance(ticker):
    return upbit.get_balance(ticker)


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

class TradingDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx+self.seq_len][['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].values
        y = self.data.iloc[idx + self.seq_len]['future_return']
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_transformer_model(ticker, epochs=100):
    print(f"모델 학습 시작: {ticker}")
    data = get_features(ticker)

    if data is None or len(data) == 0:
        print(f"경고: {ticker}의 데이터가 비어 있음. 학습 건너뜀.")
        return None

    seq_len = 20  # 시퀀스 길이 최적화
    dataset = TradingDataset(data, seq_len)

    if len(dataset) == 0:
        print(f"경고: {ticker}의 데이터셋이 너무 작음.")
        return None

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=False)  # num_workers 조정

    input_dim = 6
    d_model = 64  # 축소
    num_heads = 4  # 축소
    num_layers = 1  # 축소
    output_dim = 1

    model = TransformerModel(input_dim, d_model, num_heads, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # CPU만 사용하도록 설정
    device = torch.device("cpu")  # CUDA 대신 CPU 사용
    model.to(device)
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0  # 각 에폭의 손실 기록
        model.train()

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}')  # 평균 손실 출력

    print(f"모델 학습 완료: {ticker}")
    return model
  
def get_ml_signal(ticker, model):
    """AI 신호 계산"""
    try:
        # 특징 데이터 가져오기
        features = get_features(ticker)

        # 필요한 열만 추출하고, 마지막 30개 데이터 선택
        latest_data = features[['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].tail(20)
        
        # 데이터 정규화 (스케일링)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(latest_data)

        # 텐서로 변환 (모델에 입력, float64 사용)
        X_latest = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)

        # 모델 평가 모드로 전환
        model.eval()

        # 예측 계산
        with torch.no_grad():
            prediction = model(X_latest).item()

        # 예측값 반환
        return prediction

    except Exception as e:
        print(f"[{ticker}] AI 신호 계산 에러: {e}")
        return 0

def should_sell(ticker, current_price):
    """트레일링 스탑 로직을 활용한 매도 판단"""
    if ticker not in entry_prices:
        return False
    
    entry_price = entry_prices[ticker]
    highest_prices[ticker] = max(highest_prices[ticker], current_price)
    peak_drop = (highest_prices[ticker] - current_price) / highest_prices[ticker]

    # 동적 손절 & 익절 조건
    if peak_drop > 0.02:  # 고점 대비 2% 하락 시 익절
        return True
    elif (current_price - entry_price) / entry_price < STOP_LOSS_THRESHOLD:
        return True  # 손절 조건

    return False
    
def backtest(ticker, model, initial_balance=1_000_000, fee=0.0005):
    """과거 데이터로 백테스트 실행"""
    data = get_features(ticker)
    balance = initial_balance
    position = 0
    entry_price = 0

    for i in range(50, len(data) - 1):
        x_input = torch.tensor(data.iloc[i-30:i][['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].values,
                               dtype=torch.float32).unsqueeze(0)
        signal = model(x_input).item()

        current_price = data.iloc[i]['close']

        if position == 0 and signal > ML_THRESHOLD:
            position = balance / current_price
            entry_price = current_price
            balance = 0

        elif position > 0 and should_sell(ticker, current_price):
            balance = position * current_price * (1 - fee)
            position = 0

    final_value = balance + (position * data.iloc[-1]['close'])
    return final_value / initial_balance
    
if __name__ == "__main__":
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    print("자동매매 시작!")

    tickers = pyupbit.get_tickers(fiat="KRW")
    models = {}

    # 초기 설정
    top_tickers = get_top_tickers(n=10)
    print(f"거래량 상위 코인: {top_tickers}")
    models = {ticker: train_transformer_model(ticker) for ticker in top_tickers}
    recent_surge_tickers = {}  # 급상승 코인 저장

    try:
        while True:
            now = datetime.now()

            # ✅ 1. 상위 코인 업데이트 (6시간마다)
            if now.hour % 6 == 0 and now.minute == 0:
                top_tickers = get_top_tickers(n=5)
                print(f"[{now}] 상위 코인 업데이트: {top_tickers}")

                # 새롭게 추가된 코인 모델 학습
                for ticker in top_tickers:
                    if ticker not in models:
                        models[ticker] = train_transformer_model(ticker)

            # ✅ 2. 급상승 코인 감지 및 업데이트
            surge_tickers = detect_surge_tickers(threshold=0.03)

            # 📌 급상승 코인 저장 및 모델 학습
            for ticker in surge_tickers:
                if ticker not in recent_surge_tickers:
                    print(f"[{now}] 급상승 감지: {ticker}")
                    recent_surge_tickers[ticker] = now
                    if ticker not in models:
                        models[ticker] = train_transformer_model(ticker, epochs=10)

            # ✅ 3. 최종 매수 대상 선정 (상위 10개 + 급상승 코인 포함)
            target_tickers = set(top_tickers) | set(recent_surge_tickers.keys())  # 🔥 급상승 코인 확실히 포함!

            for ticker in target_tickers:
                last_trade_time = recent_trades.get(ticker, datetime.min)
                cooldown_limit = SURGE_COOLDOWN_TIME if ticker in recent_surge_tickers else COOLDOWN_TIME

                # ✅ [쿨다운 적용] 너무 빠른 재거래 방지
                if now - last_trade_time < cooldown_limit:
                    continue  

                try:
                    # 🔍 AI 및 지표 계산
                    ml_signal = get_ml_signal(ticker, models[ticker])
                    macd, signal = get_macd(ticker)
                    rsi = get_rsi(ticker)
                    adx = get_adx(ticker)
                    current_price = pyupbit.get_current_price(ticker)

                    # 🛠 [DEBUG] 로그 추가
                    print(f"[DEBUG] {ticker} 매수 조건 검사")
                    print(f" - ML 신호: {ml_signal:.4f}")
                    print(f" - MACD: {macd:.4f}, Signal: {signal:.4f}")
                    print(f" - RSI: {rsi:.2f}")
                    print(f" - ADX: {adx:.2f}")
                    print(f" - 현재 가격: {current_price:.2f}")

                    # ✅ 4. 매수 조건 검사 (급상승 포함)
                    if isinstance(ml_signal, (int, float)) and 0 <= ml_signal <= 1:
                        if ml_signal > ML_THRESHOLD and macd > signal and rsi < 40 and adx > 20:
                            krw_balance = get_balance("KRW")
                            print(f"[DEBUG] 보유 원화 잔고: {krw_balance:.2f}")
                            if krw_balance > 5000:
                                buy_amount = krw_balance * 0.3
                                buy_result = buy_crypto_currency(ticker, buy_amount)
                                if buy_result:
                                    entry_prices[ticker] = current_price
                                    highest_prices[ticker] = current_price
                                    recent_trades[ticker] = now
                                    print(f"[{ticker}] 매수 완료: {buy_amount:.2f}원, 가격: {current_price:.2f}")
                                else:
                                    print(f"[{ticker}] 매수 요청 실패")
                            else:
                                print(f"[{ticker}] 매수 불가 (원화 부족)")
                        else:
                            print(f"[{ticker}] 매수 조건 불충족")

                    # ✅ 5. 매도 조건 검사
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
                                coin_balance = get_balance(ticker)
                                if coin_balance > 0:
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
