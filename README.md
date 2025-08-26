# ☁️Sign Language AI Inference Server
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
©2025

# ✋ → 💬 1. Video to Gloss Model (FastAPI)
Docker 및 **FastAPI**를 기반으로 구현한 수어 인식 AI 추론 서버입니다.
개별 프레임 전송 방식(슬라이딩 윈도우) 및 `.npy` 업로드 방식을 지원합니다.

---

## 📂 1.1. 프로젝트 구조

```
sign-docker-api/
├─ app/
│  ├─ main.py                # FastAPI 서버 코드
│  └─ requirements.txt       # Python 의존성
├─ model/
│  ├─ frame_to_gloss_v0.h5   # 학습된 모델
│  └─ frame_to_gloss_v0.json # 라벨 매핑
├─ Dockerfile
└─ docker-compose.yml
```

---
## 🚀 1.2. 실행 방법
### 1.2.1. Docker 설치

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치
* 설치 후 터미널에서 확인:

```bash
docker --version
docker compose version
```

### 1.2.2. 레포지토리 클론

```bash
git clone <repository-url>
cd sign-docker-api
```

### 1.2.3. 빌드 & 실행

```bash
docker compose build --no-cache
docker compose up
```

* 서버가 실행되면:

```
Uvicorn running on http://0.0.0.0:8000
```

### 1.2.4. 종료

```bash
docker compose down
```
---

## 🌐 1.3. API 엔드포인트

### 1.3.1. 헬스체크

* **GET** `/health`
  서버 상태 및 설정 확인

```json
{
  "status": "ok",
  "window": 10,
  "features": 194,
  "sessions": 0
}
```

### 1.3.2. 프레임 개별 전송 (슬라이딩 윈도우)

* **POST** `/predict/frame`
* **Request Body (JSON)**:

```json
{
  "session_id": "user-or-device-uuid",
  "keypoints": [0.12, 0.03, "..."]  // 길이 194
}
```

* **Response**:

  * 수집 중:

    ```json
    { "status": "collecting", "collected": 7, "window": 10 }
    ```
  * 예측 완료:

    ```json
    { "label": "지시1#", "confidence": 0.87, "window": 10 }
    ```

### 1.3.3. NPY 파일 업로드 예측

* **POST** `/predict/npy`
* **Form Data**:

  * file: `.npy` 파일
* **Response**:

```json
{ "label": "지시1#", "confidence": 0.95 }
```

### 1.3.4. 세션 초기화

* **DELETE** `/predict/session/{sid}`

### 1.3.5. 오래된 세션 정리

* **DELETE** `/predict/sessions/cleanup`

---

## 🧪 1.4. API 테스트 예제

### 헬스체크

```bash
curl http://localhost:8000/health
```

### 프레임 개별 전송

```bash
curl -X POST "http://localhost:8000/predict/frame" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test123", "keypoints":[0.1,0.2,...]}'
```

### NPY 업로드

```bash
curl -X POST "http://localhost:8000/predict/npy" \
  -F "file=@sample.npy"
```

---

## 📌 1.5. 주의사항
* **WINDOW**, **FEATURES**, **CONF\_THRESHOLD** 값은 서버와 클라이언트 모두 동일해야 합니다.
* `session_id`는 각 사용자/기기별로 고유해야 합니다.
* `.npy` 데이터는 `(frames, features)` 형태여야 하며, features는 194로 고정됩니다.
* 로컬이 아닌 외부에서 접속하려면, 서버 IP 또는 도메인을 사용하고 포트를 개방해야 합니다.
* Docker 컨테이너 실행 시 모델 파일(`.h5`, `.json`)이 `/model` 경로에 존재해야 하며, 변경 시 재빌드 필요
---
# 💬 → ✋ 2.Text to Sign Language Model (FlasKAPI)

Docker 및 **Flask**를 기반으로 구현한 Text to gloss AI 추론 서버입니다.
Text를 get 방식으로 요청시 해당 Text를 gloss로 변환하여 gloss에 대응되는 수어의 동작을 Mediapipe 변환된 좌표값으로 제공된다.  

---
## 📂 2.1. 프로젝트 구조
```
T2G_flask/
├── app.py              # 1. Flask 애플리케이션 실행 및 API 엔드포인트 정의
├── inference.py        # 2. 텍스트를 수어 '글로스(Gloss)'로 변환 (모델 추론)
├── mapping_point.py    # 3. 변환된 '글로스'를 실제 좌표 데이터(point)로 매핑
├── data_load.py        # 4. 모델이 사용하는 데이터(유효 글로스 목록) 로드
├── download_from_s3.py # 5. S3에서 모델 및 필요 에셋 다운로드
├── config.py           # 6. S3 경로, 로컬 경로 등 주요 설정 관리
├── Dockerfile          # 7. 애플리케이션 배포를 위한 Docker 이미지 설정
├── environment.final.yml # 8. Docker 컨테이너의 Conda 환경 정의 (핵심 의존성)
└── test.py         
```
## 🚀 2.2. 실행 방법
### 2.2.1. Docker 설치
- 1.2.1번과 동일

### 2.2.2. Docker 빌드 및 실행
```bash
docker run -d -p 1958:1958 --name kseb-t2g-server -e AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY hahagoodgood/flask-kseb-t2g
```

### 2.2.3. Docker 컨테이너 종료
```bash
docker stop kseb-t2g-server
```

## 🌐 2.3. API 엔드포인트
### 2.3.1. 헬스체크

* **GET** `/`
  서버 상태 및 설정 확인

* **Response**:

  * 정상(200):

    ```json
    {"status": "Service is running"}
    ```
  * 오류(503):

    ```json
    {"error": "Model pipeline is not available. The service is likely initializing or has failed."}
    ```

### 2.3.2. Gloss 추론 후 Body 좌표값 전달

* **GET** `/T2G/translate`
* **parameter**:
   - text : 번역을 요청하는 문장

* **Response**:

  * 출력(200):

    ```json
    [
      {"left_hand": [
          {
            "name": "WRIST",
            "x": 0.5,
            "y": 0.6,
            "z": -0.4
          },
          "..."
        ],
        "right_hand": [
          {
            "name": "WRIST",
            "x": 0.4,
            "y": 0.6,
            "z": -0.35
          },
          "..."
        ],
        "pose": [
          {
            "name": "NOSE",
            "x": 0.45,
            "y": 0.2,
            "z": -0.8,
            "visibility": 0.98
          },
          "..."
        ]
      },
      {
        "left_hand": "[...]",
        "right_hand": "[...]",
        "pose": "[...]"
      },
      "..."
    ]
    ```
  * 모델 로드 불가(503):

    ```json
    {"error": "Model pipeline is not available."}
    ```
    
  * 'text' 파라미터 없이 요청시 (400):
    ```json
    {"error": "번역할 'text' 파라미터가 필요합니다."}
    ```
  * error(500):
    ```json
    {"error": "번역 중 오류가 발생했습니다."}
    ```
## 🧪 2.4. API 테스트 예제
### 헬스체크
>http://127.0.0.1:1958/

### 프레임 개별 전송
>http://127.0.0.1:1958//T2G/translate?text=사춘기 때 아이에게 일어나는 변화를 잘 이해하고 지나가는 것이 필요합니다.