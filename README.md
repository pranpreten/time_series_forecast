# 미세먼지 시계열 예측 웹 데모

도시별 PM10·PM2.5 데이터를 전처리하고 앙상블 회귀 모델로 예측한 결과를 Flask 웹 화면에 제공하는 프로젝트다. 데이터 생성, 특징 구성, 학습, 모델 저장, 웹 추론의 흐름을 구현한다.

## 구현 범위

- 도시와 시간에 따른 대기질·기상 변수의 합성 데이터 생성
- 시차·이동 통계·주기 특성 생성 및 스케일링
- Random Forest, XGBoost, LightGBM, Ridge 기반 앙상블 학습
- PM10·PM2.5 모델과 전처리 객체 저장
- 도시별 현재 값, 과거 기록 및 향후 24시간 예측 API 제공

`data_collector.py`는 실제 관측 API를 호출하는 수집기가 아니라 합성 데이터를 만드는 스크립트다. 화면의 현재 값과 예측도 저장된 데이터에 기반한 데모이며 실시간 관측 정보가 아니다.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `data_collector.py` | 시계열 합성 데이터 생성 |
| `preprocessing.py` | 시차·이동 통계·주기 특성과 스케일러 구성 |
| `train_model.py` | 모델 학습·평가 및 저장 |
| `app.py` | Flask 화면과 추론 API |
| `data/air_quality.csv` | 학습·시연용 데이터 |
| `models/` | PM10·PM2.5 예측기와 스케일러 |
| `templates/`, `static/` | 웹 화면과 스타일 |

## 실행 방법

저장소 루트에서 실행한다. 고정 의존성에 맞는 Python 환경이 필요하며 Python 3.10 또는 3.11 환경을 준비하는 것이 적절하다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

브라우저에서 `http://localhost:5000`에 접속한다. 저장소의 `data/`와 `models/`가 준비되어 있으면 기존 모델로 실행한다.

데이터와 모델을 새로 만들려면 다음 순서로 실행한다. 기존 CSV와 모델 파일이 갱신된다.

```powershell
.\.venv\Scripts\python.exe data_collector.py
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe app.py
```

## 결과 확인 및 한계

학습 스크립트는 MAE·RMSE·R²를 출력하고 웹 화면은 도시별 대기질 변화와 예측을 보여준다. 합성 데이터에서 계산한 지표를 실제 대기질 예측 성능으로 해석하지 않는다. 실제 서비스로 확장하려면 관측 데이터 연동, 시간순 검증, 예보 시점에 사용 가능한 입력값의 정합성을 별도로 확인해야 한다.

---

[GitHub 저장소](https://github.com/pranpreten/time_series_forecast) · [문서 작성 기준 코드](https://github.com/pranpreten/time_series_forecast/tree/ef40aa7ad337758cab2ee99d6f27755e28f2917b)
