"""
미세먼지 데이터 전처리 모듈
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def load_data(filepath='data/air_quality.csv'):
    """CSV 데이터 로드"""
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return df


def create_lag_features(df, target_col, lags=[1, 2, 3, 6, 12, 24]):
    """시계열 lag 특성 생성"""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby('city')[target_col].shift(lag)
    return df


def create_rolling_features(df, target_col, windows=[6, 12, 24]):
    """이동평균 특성 생성"""
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = (
            df.groupby('city')[target_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f'{target_col}_rolling_std_{window}'] = (
            df.groupby('city')[target_col]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
    return df


def encode_cyclical(df, col, max_val):
    """순환 특성 인코딩 (시간, 요일, 월)"""
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def preprocess_data(df, target='pm10', is_training=True, scalers=None):
    """
    전체 전처리 파이프라인

    Args:
        df: 원본 데이터프레임
        target: 예측 대상 ('pm10' 또는 'pm25')
        is_training: 학습 모드 여부
        scalers: 저장된 스케일러 (예측 시 사용)

    Returns:
        X, y, scalers (학습 시) 또는 X, scalers (예측 시)
    """
    df = df.copy()

    # 도시 인코딩
    if is_training:
        city_encoder = LabelEncoder()
        df['city_encoded'] = city_encoder.fit_transform(df['city'])
    else:
        city_encoder = scalers['city_encoder']
        df['city_encoded'] = city_encoder.transform(df['city'])

    # 시간 특성 순환 인코딩
    df = encode_cyclical(df, 'hour', 24)
    df = encode_cyclical(df, 'weekday', 7)
    df = encode_cyclical(df, 'month', 12)

    # 풍향 순환 인코딩
    df = encode_cyclical(df, 'wind_direction', 360)

    # Lag 특성 생성
    df = create_lag_features(df, target, lags=[1, 2, 3, 6, 12, 24])

    # 이동평균 특성 생성
    df = create_rolling_features(df, target, windows=[6, 12, 24])

    # 결측치 제거 (lag로 인한)
    df = df.dropna()

    # 특성 선택
    feature_cols = [
        'city_encoded',
        'temperature', 'humidity', 'wind_speed', 'precipitation',
        'hour_sin', 'hour_cos',
        'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos',
        'wind_direction_sin', 'wind_direction_cos',
        f'{target}_lag_1', f'{target}_lag_2', f'{target}_lag_3',
        f'{target}_lag_6', f'{target}_lag_12', f'{target}_lag_24',
        f'{target}_rolling_mean_6', f'{target}_rolling_mean_12', f'{target}_rolling_mean_24',
        f'{target}_rolling_std_6', f'{target}_rolling_std_12', f'{target}_rolling_std_24',
    ]

    X = df[feature_cols].values

    # 스케일링
    if is_training:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        scalers = {
            'feature_scaler': scaler,
            'city_encoder': city_encoder,
            'feature_cols': feature_cols
        }
        y = df[target].values
        return X, y, scalers, df
    else:
        scaler = scalers['feature_scaler']
        X = scaler.transform(X)
        return X, scalers, df


def prepare_future_features(current_data, city, hours_ahead=24):
    """
    미래 예측을 위한 특성 준비
    현재 데이터를 기반으로 미래 시점의 특성 생성
    """
    future_data = []
    last_row = current_data[current_data['city'] == city].iloc[-1].copy()

    for h in range(1, hours_ahead + 1):
        new_row = last_row.copy()
        future_time = pd.Timestamp(last_row['timestamp']) + pd.Timedelta(hours=h)

        new_row['timestamp'] = future_time
        new_row['hour'] = future_time.hour
        new_row['weekday'] = future_time.weekday()
        new_row['month'] = future_time.month

        # 기상 데이터는 약간의 변동 추가 (실제로는 기상 예보 사용)
        new_row['temperature'] += np.random.uniform(-2, 2)
        new_row['humidity'] += np.random.uniform(-5, 5)
        new_row['humidity'] = np.clip(new_row['humidity'], 20, 100)
        new_row['wind_speed'] += np.random.uniform(-1, 1)
        new_row['wind_speed'] = max(0, new_row['wind_speed'])

        future_data.append(new_row)

    return pd.DataFrame(future_data)


def save_scalers(scalers, filepath='models/scalers.pkl'):
    """스케일러 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scalers, filepath)
    print(f"스케일러 저장 완료: {filepath}")


def load_scalers(filepath='models/scalers.pkl'):
    """스케일러 로드"""
    return joblib.load(filepath)


if __name__ == '__main__':
    # 테스트
    df = load_data()
    print(f"원본 데이터: {len(df)} 레코드")

    X, y, scalers, processed_df = preprocess_data(df, target='pm10')
    print(f"전처리 후: {X.shape}")
    print(f"특성 개수: {X.shape[1]}")
    print(f"특성 목록: {scalers['feature_cols']}")
