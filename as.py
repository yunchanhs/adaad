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
import concurrent.futures
from torch.optim.lr_scheduler import ReduceLROnPlateau
import multiprocessing as mp
import threading

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# 모델 학습 주기 관련 변수
last_trained_time = None  # 마지막 학습 시간
TRAINING_INTERVAL = timedelta(hours=6)  # 6시간마다 재학습

# 매매 전략 관련 임계값
ML_THRESHOLD = 0.01  # AI 신호 매수 기준
ML_SELL_THRESHOLD = 0.3  # AI 신호 매도 기준
STOP_LOSS_THRESHOLD = -0.05  # 손절 (-5%)
TAKE_PROFIT_THRESHOLD = 0.1  # 익절 (10%)
COOLDOWN_TIME = timedelta(minutes=5)  # 동일 코인 재거래 쿨다운 시간

# 계좌 정보 저장
entry_prices = {}  # 매수한 가격 저장
highest_prices = {}  # 매수 후 최고 가격 저장
recent_trades = {}  # 최근 거래 기록
recent_surge_tickers = {}  # 최근 급상승 감지 코인 저장

def get_top_tickers(n=30):
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
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1  # 미래 수익률

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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 평균 풀링
        return self.fc_out(x)

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

def train_transformer_model(ticker, epochs=50):
    print(f"모델 학습 시작: {ticker}")
    data = get_features(ticker)
    seq_len = 30
    dataset = TradingDataset(data, seq_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    input_dim = 6
    d_model = 64
    num_heads = 8
    num_layers = 4
    output_dim = 1

    model = TransformerModel(input_dim, d_model, num_heads, num_layers, output_dim)
    criterion = nn.HuberLoss(delta=1.0)  # Huber 손실 함수 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.view(-1), y_batch.view(-1))  # `.squeeze()` 대신 `.view(-1)`
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
        
        # Transformer 입력 차원 맞추기 (필요할 경우)
        if X_latest.shape[-1] != model.embedding.in_features:
            print(f"[경고] 입력 차원 불일치: {X_latest.shape} vs {model.embedding.in_features}")

        model.eval()
        with torch.no_grad():
            prediction = model(X_latest).item()
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

    for i in range(30, len(data) - 1):
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

# ✅ 모델 학습을 병렬 처리하는 함수 (멀티프로세싱)
def train_transformer_model_mp(ticker):
    return ticker, train_transformer_model(ticker)

def train_models_parallel(tickers):
    with mp.Pool(processes=min(len(tickers), mp.cpu_count())) as pool:
        results = pool.map(train_transformer_model_mp, tickers)
    return {ticker: model for ticker, model in results}

# ✅ 매매 로직을 처리하는 스레드 함수 (멀티스레딩)
def trading_thread(ticker, models, recent_trades, recent_surge_tickers, entry_prices, highest_prices):
    while True:
        now = datetime.now()
        last_trade_time = recent_trades.get(ticker, datetime.min)
        cooldown_limit = SURGE_COOLDOWN_TIME if ticker in recent_surge_tickers else COOLDOWN_TIME

        # 쿨다운 적용
        if now - last_trade_time < cooldown_limit:
            time.sleep(1)
            continue

        try:
            # 🔍 AI 및 지표 계산
            ml_signal = get_ml_signal(ticker, models[ticker])
            macd, signal = get_macd(ticker)
            rsi = get_rsi(ticker)
            adx = get_adx(ticker)
            current_price = pyupbit.get_current_price(ticker)

            print(f"[DEBUG] {ticker} 매수 조건 검사")
            print(f" - ML 신호: {ml_signal:.4f}")
            print(f" - MACD: {macd:.4f}, Signal: {signal:.4f}")
            print(f" - RSI: {rsi:.2f}")
            print(f" - ADX: {adx:.2f}")
            print(f" - 현재 가격: {current_price:.2f}")

            # ✅ 매수 조건 검사 (급상승 포함)
            if isinstance(ml_signal, (int, float)) and 0 <= ml_signal <= 1:
                if ml_signal > ML_THRESHOLD and macd >= signal and rsi < 50 and adx > 15:
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

            # ✅ 매도 조건 검사
            elif ticker in entry_prices:
                entry_price = entry_prices[ticker]
                highest_prices[ticker] = max(highest_prices[ticker], current_price)
                change_ratio = (current_price - entry_price) / entry_price

                # 🛠  [DEBUG] 매도 조건 확인용 로그 추가
                print(f"[DEBUG] {ticker} 매도 조건 검사")
                print(f" - 진입 가격: {entry_price:.2f}")
                print(f" - 최고 가격: {highest_prices[ticker]:.2f}")
                print(f" - 현재 가격: {current_price:.2f}")
                print(f" - 변동률: {change_ratio:.4f}")
                print(f" - AI 신호: {ml_signal:.4f}")

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

        time.sleep(1)

if __name__ == "__main__":
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    print("자동매매 시작!")

    tickers = pyupbit.get_tickers(fiat="KRW")
    models = {}

    # ✅ 기존 학습 방식 유지 (상위 10개 코인)
    top_tickers = get_top_tickers(n=30)
    print(f"거래량 상위 코인: {top_tickers}")

    # ✅ 기존 모델 학습 (병렬 처리)
    models = {ticker: train_transformer_model(ticker) for ticker in top_tickers}

    # 📌 급상승 코인 저장
    recent_surge_tickers = {}  

    try:
        while True:
            now = datetime.now()

            # ✅ 1. 상위 코인 업데이트 (6시간마다)
            if now.hour % 6 == 0 and now.minute == 0:
                top_tickers = get_top_tickers(n=30)
                print(f"[{now}] 상위 코인 업데이트: {top_tickers}")

                # 새롭게 추가된 코인 모델 학습 (병렬 처리)
                new_tickers = [ticker for ticker in top_tickers if ticker not in models]
                if new_tickers:
                    models.update(train_models_parallel(new_tickers))  # 모델 학습 병렬 처리

            # ✅ 2. 급상승 코인 감지 및 업데이트
            surge_tickers = detect_surge_tickers(threshold=0.03)

            # 📌 병렬 학습 적용
            new_surge_tickers = [ticker for ticker in surge_tickers if ticker not in recent_surge_tickers]
            if new_surge_tickers:
                print(f"[{now}] 병렬 학습 시작: {new_surge_tickers}")
                new_models = train_models_parallel(new_surge_tickers)  # 모델 학습 병렬 처리
                models.update(new_models)  # 기존 모델에 추가
                recent_surge_tickers.update({ticker: now for ticker in new_surge_tickers})

            # ✅ 3. 매수 대상 선정 (상위 10개 + 급상승 코인 포함)
            target_tickers = set(top_tickers) | set(recent_surge_tickers.keys())

            # ✅ 매매 로직을 위한 스레드 실행
            threads = []
            for ticker in target_tickers:
                thread = threading.Thread(target=trading_thread, args=(ticker, models, recent_trades, recent_surge_tickers, entry_prices, highest_prices))
                thread.daemon = True  # 메인 종료 시 스레드 종료
                thread.start()
                threads.append(thread)

            # 스레드가 실행 중인 동안 메인 루프는 잠시 대기
            time.sleep(1)

    except KeyboardInterrupt:
        print("프로그램이 종료되었습니다.")
