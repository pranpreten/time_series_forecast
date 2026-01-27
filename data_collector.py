"""
미세먼지 데이터 수집/생성 모듈
실제 한국 미세먼지 패턴을 반영한 합성 데이터 생성
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 대한민국 주요 도시 정보
CITIES = {
    '서울': {'lat': 37.5665, 'lon': 126.9780, 'base_pm10': 45, 'base_pm25': 25},
    '부산': {'lat': 35.1796, 'lon': 129.0756, 'base_pm10': 38, 'base_pm25': 20},
    '대구': {'lat': 35.8714, 'lon': 128.6014, 'base_pm10': 42, 'base_pm25': 23},
    '인천': {'lat': 37.4563, 'lon': 126.7052, 'base_pm10': 48, 'base_pm25': 27},
    '광주': {'lat': 35.1595, 'lon': 126.8526, 'base_pm10': 35, 'base_pm25': 18},
    '대전': {'lat': 36.3504, 'lon': 127.3845, 'base_pm10': 40, 'base_pm25': 22},
    '울산': {'lat': 35.5384, 'lon': 129.3114, 'base_pm10': 43, 'base_pm25': 24},
}


def generate_air_quality_data(days=365):
    """
    실제 미세먼지 패턴을 반영한 합성 데이터 생성
    - 계절성: 봄(3-5월) 황사, 겨울(12-2월) 난방
    - 일간 패턴: 출퇴근 시간 증가
    - 기상 영향: 풍속, 습도, 온도
    """
    np.random.seed(42)

    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    current_date = start_date
    while current_date <= end_date:
        for city, info in CITIES.items():
            for hour in range(24):
                timestamp = current_date.replace(hour=hour, minute=0, second=0)

                # 기본값
                base_pm10 = info['base_pm10']
                base_pm25 = info['base_pm25']

                # 계절성 (월 기준)
                month = current_date.month
                if month in [3, 4, 5]:  # 봄 - 황사
                    seasonal_factor = 1.4 + np.random.uniform(0, 0.3)
                elif month in [12, 1, 2]:  # 겨울 - 난방
                    seasonal_factor = 1.3 + np.random.uniform(0, 0.2)
                elif month in [6, 7, 8]:  # 여름 - 비로 청정
                    seasonal_factor = 0.7 + np.random.uniform(0, 0.2)
                else:  # 가을
                    seasonal_factor = 0.9 + np.random.uniform(0, 0.1)

                # 일간 패턴 (출퇴근 시간)
                if hour in [7, 8, 9, 18, 19, 20]:
                    hourly_factor = 1.2 + np.random.uniform(0, 0.1)
                elif hour in [2, 3, 4, 5]:
                    hourly_factor = 0.7 + np.random.uniform(0, 0.1)
                else:
                    hourly_factor = 1.0 + np.random.uniform(-0.1, 0.1)

                # 요일 패턴 (주말 감소)
                weekday = current_date.weekday()
                if weekday >= 5:  # 주말
                    weekday_factor = 0.85
                else:
                    weekday_factor = 1.0

                # 기상 데이터 생성
                # 온도 (계절별)
                if month in [12, 1, 2]:
                    temperature = np.random.uniform(-10, 5)
                elif month in [3, 4, 5]:
                    temperature = np.random.uniform(5, 20)
                elif month in [6, 7, 8]:
                    temperature = np.random.uniform(22, 35)
                else:
                    temperature = np.random.uniform(10, 22)

                # 습도
                humidity = np.random.uniform(30, 90)

                # 풍속 (강할수록 미세먼지 감소)
                wind_speed = np.random.exponential(2.5)
                wind_speed = min(wind_speed, 15)
                wind_factor = max(0.5, 1 - wind_speed * 0.03)

                # 풍향 (서풍일 때 중국발 미세먼지 증가)
                wind_direction = np.random.uniform(0, 360)
                if 225 <= wind_direction <= 315:  # 서풍
                    wind_dir_factor = 1.2
                else:
                    wind_dir_factor = 1.0

                # 강수량 (비오면 미세먼지 감소)
                if np.random.random() < 0.15:  # 15% 확률로 비
                    precipitation = np.random.exponential(5)
                    precip_factor = max(0.3, 1 - precipitation * 0.05)
                else:
                    precipitation = 0
                    precip_factor = 1.0

                # 최종 미세먼지 계산
                pm10 = base_pm10 * seasonal_factor * hourly_factor * weekday_factor
                pm10 = pm10 * wind_factor * wind_dir_factor * precip_factor
                pm10 += np.random.normal(0, 10)
                pm10 = max(5, pm10)

                pm25 = base_pm25 * seasonal_factor * hourly_factor * weekday_factor
                pm25 = pm25 * wind_factor * wind_dir_factor * precip_factor
                pm25 += np.random.normal(0, 5)
                pm25 = max(2, pm25)

                data.append({
                    'timestamp': timestamp,
                    'city': city,
                    'pm10': round(pm10, 1),
                    'pm25': round(pm25, 1),
                    'temperature': round(temperature, 1),
                    'humidity': round(humidity, 1),
                    'wind_speed': round(wind_speed, 1),
                    'wind_direction': round(wind_direction, 1),
                    'precipitation': round(precipitation, 1),
                    'weekday': weekday,
                    'hour': hour,
                    'month': month
                })

        current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    return df


def save_data(df, filepath='data/air_quality.csv'):
    """데이터를 CSV로 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"데이터 저장 완료: {filepath}")
    print(f"총 {len(df):,}개 레코드")
    print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"도시: {df['city'].unique().tolist()}")


if __name__ == '__main__':
    print("미세먼지 데이터 생성 중...")
    df = generate_air_quality_data(days=365)
    save_data(df)
    print("\n데이터 샘플:")
    print(df.head(10))
