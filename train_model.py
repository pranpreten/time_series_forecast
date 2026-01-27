"""
미세먼지 예측 모델 학습 모듈
RandomForest, XGBoost, LightGBM 앙상블
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

from data_collector import generate_air_quality_data, save_data, CITIES
from preprocessing import load_data, preprocess_data, save_scalers


def train_ensemble_model(X_train, y_train):
    """앙상블 모델 학습"""
    print("모델 학습 시작...")

    # 개별 모델 정의
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    ridge_model = Ridge(alpha=1.0)

    # 앙상블 (Voting Regressor)
    ensemble = VotingRegressor(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('ridge', ridge_model)
        ],
        n_jobs=-1
    )

    ensemble.fit(X_train, y_train)
    print("모델 학습 완료!")

    return ensemble


def evaluate_model(model, X_test, y_test):
    """모델 평가"""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== 모델 평가 결과 =====")
    print(f"MAE (평균 절대 오차): {mae:.2f}")
    print(f"RMSE (평균 제곱근 오차): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def save_model(model, filepath='models/pm_predictor.pkl'):
    """모델 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"모델 저장 완료: {filepath}")


def load_model(filepath='models/pm_predictor.pkl'):
    """모델 로드"""
    return joblib.load(filepath)


def train_both_models():
    """PM10과 PM2.5 모델 모두 학습"""

    # 데이터 생성 (없으면)
    if not os.path.exists('data/air_quality.csv'):
        print("데이터 생성 중...")
        df = generate_air_quality_data(days=365)
        save_data(df)
    else:
        print("기존 데이터 로드...")
        df = load_data()

    results = {}

    for target in ['pm10', 'pm25']:
        print(f"\n{'='*50}")
        print(f"{target.upper()} 모델 학습")
        print('='*50)

        # 전처리
        X, y, scalers, processed_df = preprocess_data(df, target=target)

        # 학습/테스트 분리 (시계열이므로 셔플 안함)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"학습 데이터: {X_train.shape[0]:,}개")
        print(f"테스트 데이터: {X_test.shape[0]:,}개")

        # 모델 학습
        model = train_ensemble_model(X_train, y_train)

        # 평가
        metrics = evaluate_model(model, X_test, y_test)
        results[target] = metrics

        # 저장
        save_model(model, f'models/{target}_predictor.pkl')
        save_scalers(scalers, f'models/{target}_scalers.pkl')

    return results


if __name__ == '__main__':
    results = train_both_models()

    print("\n" + "="*50)
    print("최종 결과 요약")
    print("="*50)
    for target, metrics in results.items():
        print(f"\n{target.upper()}:")
        print(f"  MAE: {metrics['mae']:.2f} µg/m³")
        print(f"  RMSE: {metrics['rmse']:.2f} µg/m³")
        print(f"  R²: {metrics['r2']:.4f}")
