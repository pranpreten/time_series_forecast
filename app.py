"""
미세먼지 예측 Flask 웹 애플리케이션
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# 도시 정보
CITIES = ['서울', '부산', '대구', '인천', '광주', '대전', '울산']

# 대기질 등급 기준
def get_air_quality_grade(pm10, pm25):
    """대기질 등급 계산"""
    # PM10 기준
    if pm10 <= 30:
        pm10_grade = ('좋음', 'good', 1)
    elif pm10 <= 80:
        pm10_grade = ('보통', 'moderate', 2)
    elif pm10 <= 150:
        pm10_grade = ('나쁨', 'bad', 3)
    else:
        pm10_grade = ('매우나쁨', 'very-bad', 4)

    # PM2.5 기준
    if pm25 <= 15:
        pm25_grade = ('좋음', 'good', 1)
    elif pm25 <= 35:
        pm25_grade = ('보통', 'moderate', 2)
    elif pm25 <= 75:
        pm25_grade = ('나쁨', 'bad', 3)
    else:
        pm25_grade = ('매우나쁨', 'very-bad', 4)

    # 더 나쁜 등급 반환
    if pm10_grade[2] >= pm25_grade[2]:
        return pm10_grade
    return pm25_grade


def load_models():
    """모델과 스케일러 로드"""
    models = {}
    scalers = {}

    for target in ['pm10', 'pm25']:
        model_path = f'models/{target}_predictor.pkl'
        scaler_path = f'models/{target}_scalers.pkl'

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            models[target] = joblib.load(model_path)
            scalers[target] = joblib.load(scaler_path)

    return models, scalers


def load_latest_data():
    """최신 데이터 로드"""
    if os.path.exists('data/air_quality.csv'):
        df = pd.read_csv('data/air_quality.csv', parse_dates=['timestamp'])
        return df
    return None


# 모델 로드 (앱 시작 시)
models, scalers = load_models()
base_data = load_latest_data()


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html', cities=CITIES)


@app.route('/api/cities')
def get_cities():
    """도시 목록 API"""
    return jsonify({'cities': CITIES})


@app.route('/api/current/<city>')
def get_current(city):
    """현재 대기질 정보"""
    if base_data is None:
        return jsonify({'error': '데이터가 없습니다'}), 404

    city_data = base_data[base_data['city'] == city]
    if city_data.empty:
        return jsonify({'error': '도시를 찾을 수 없습니다'}), 404

    # 현재 시간 이전의 가장 최근 데이터 찾기
    now = pd.Timestamp.now()
    city_data = city_data.copy()
    past_data = city_data[city_data['timestamp'] <= now]
    latest = past_data.iloc[-1] if not past_data.empty else city_data.iloc[0]

    pm10 = float(latest['pm10'])
    pm25 = float(latest['pm25'])
    grade = get_air_quality_grade(pm10, pm25)

    return jsonify({
        'city': city,
        'timestamp': latest['timestamp'].isoformat(),
        'pm10': round(pm10, 1),
        'pm25': round(pm25, 1),
        'temperature': round(float(latest['temperature']), 1),
        'humidity': round(float(latest['humidity']), 1),
        'wind_speed': round(float(latest['wind_speed']), 1),
        'grade': grade[0],
        'grade_class': grade[1]
    })


@app.route('/api/predict/<city>')
def predict(city):
    """24시간 예측 API"""
    if not models or not scalers:
        return jsonify({'error': '모델이 로드되지 않았습니다'}), 500

    if base_data is None:
        return jsonify({'error': '데이터가 없습니다'}), 404

    city_data = base_data[base_data['city'] == city].copy()
    if city_data.empty:
        return jsonify({'error': '도시를 찾을 수 없습니다'}), 404

    predictions = []
    # 현재 시간 기준으로 예측
    now = pd.Timestamp.now()
    city_data_copy = city_data.copy()
    city_data_copy['time_diff'] = abs(city_data_copy['timestamp'] - now)
    last_row = city_data_copy.loc[city_data_copy['time_diff'].idxmin()]
    current_time = now.replace(minute=0, second=0, microsecond=0)

    # 현재 PM 값
    current_pm10 = float(last_row['pm10'])
    current_pm25 = float(last_row['pm25'])

    today = current_time.date()

    for h in range(1, 25):
        future_time = current_time + timedelta(hours=h)

        # 예측을 위한 특성 생성
        features = create_prediction_features(
            city_data, city, h, future_time,
            current_pm10, current_pm25
        )

        # PM10 예측
        pm10_pred = predict_single(features, 'pm10')
        # PM2.5 예측
        pm25_pred = predict_single(features, 'pm25')

        grade = get_air_quality_grade(pm10_pred, pm25_pred)

        # 오늘/내일 구분
        if future_time.date() == today:
            day_label = '오늘'
        else:
            day_label = '내일'
        time_label = f"{day_label} {future_time.strftime('%H:%M')}"

        predictions.append({
            'hour': h,
            'timestamp': future_time.isoformat(),
            'time_label': time_label,
            'pm10': round(pm10_pred, 1),
            'pm25': round(pm25_pred, 1),
            'grade': grade[0],
            'grade_class': grade[1]
        })

        # 다음 예측을 위해 현재 값 업데이트
        current_pm10 = pm10_pred
        current_pm25 = pm25_pred

    return jsonify({
        'city': city,
        'predictions': predictions
    })


def create_prediction_features(city_data, city, hours_ahead, future_time, current_pm10, current_pm25):
    """예측을 위한 특성 생성"""
    last_row = city_data.iloc[-1]

    # 기본 특성
    features = {
        'city': city,
        'hour': future_time.hour,
        'weekday': future_time.weekday(),
        'month': future_time.month,
        'temperature': float(last_row['temperature']) + np.random.uniform(-2, 2),
        'humidity': np.clip(float(last_row['humidity']) + np.random.uniform(-5, 5), 20, 100),
        'wind_speed': max(0, float(last_row['wind_speed']) + np.random.uniform(-1, 1)),
        'wind_direction': float(last_row['wind_direction']),
        'precipitation': 0,
        'pm10': current_pm10,
        'pm25': current_pm25
    }

    return features


def predict_single(features, target):
    """단일 예측 수행"""
    if target not in models or target not in scalers:
        return features[target]  # 모델 없으면 현재값 반환

    model = models[target]
    scaler_info = scalers[target]

    # 특성 인코딩
    city_encoder = scaler_info['city_encoder']
    feature_scaler = scaler_info['feature_scaler']

    try:
        city_encoded = city_encoder.transform([features['city']])[0]
    except:
        city_encoded = 0

    # 순환 인코딩
    hour_sin = np.sin(2 * np.pi * features['hour'] / 24)
    hour_cos = np.cos(2 * np.pi * features['hour'] / 24)
    weekday_sin = np.sin(2 * np.pi * features['weekday'] / 7)
    weekday_cos = np.cos(2 * np.pi * features['weekday'] / 7)
    month_sin = np.sin(2 * np.pi * features['month'] / 12)
    month_cos = np.cos(2 * np.pi * features['month'] / 12)
    wind_dir_sin = np.sin(2 * np.pi * features['wind_direction'] / 360)
    wind_dir_cos = np.cos(2 * np.pi * features['wind_direction'] / 360)

    current_value = features[target]

    # 특성 벡터 생성 (전처리 모듈과 동일한 순서)
    X = np.array([[
        city_encoded,
        features['temperature'],
        features['humidity'],
        features['wind_speed'],
        features['precipitation'],
        hour_sin, hour_cos,
        weekday_sin, weekday_cos,
        month_sin, month_cos,
        wind_dir_sin, wind_dir_cos,
        current_value, current_value, current_value,  # lag 1, 2, 3
        current_value, current_value, current_value,  # lag 6, 12, 24
        current_value, current_value, current_value,  # rolling mean
        1.0, 1.0, 1.0  # rolling std (임시값)
    ]])

    # 스케일링
    X_scaled = feature_scaler.transform(X)

    # 예측
    pred = model.predict(X_scaled)[0]

    # 음수 방지
    return max(0, pred)


@app.route('/api/history/<city>')
def get_history(city):
    """최근 24시간 히스토리"""
    if base_data is None:
        return jsonify({'error': '데이터가 없습니다'}), 404

    city_data = base_data[base_data['city'] == city].tail(24)

    history = []
    for _, row in city_data.iterrows():
        history.append({
            'timestamp': row['timestamp'].isoformat(),
            'time_label': row['timestamp'].strftime('%H:%M'),
            'pm10': round(float(row['pm10']), 1),
            'pm25': round(float(row['pm25']), 1)
        })

    return jsonify({
        'city': city,
        'history': history
    })


if __name__ == '__main__':
    print("="*50)
    print("미세먼지 예측 웹 서비스")
    print("="*50)
    print(f"모델 로드 상태: {'성공' if models else '실패 - 먼저 train_model.py 실행 필요'}")
    print(f"데이터 로드 상태: {'성공' if base_data is not None else '실패'}")
    print("서버 시작: http://localhost:5000")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)
