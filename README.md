# Sign Language AI Inference Server

Docker 및 **FastAPI**를 기반으로 구현한 수어 인식 AI 추론 서버입니다.
개별 프레임 전송 방식(슬라이딩 윈도우) 및 `.npy` 업로드 방식을 지원합니다.

---

## 📂 프로젝트 구조

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

## 🚀 실행 방법

### 1. Docker 설치

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치
* 설치 후 터미널에서 확인:

```bash
docker --version
docker compose version
```

### 2. 레포지토리 클론

```bash
git clone <repository-url>
cd sign-docker-api
```

### 3. 빌드 & 실행

```bash
docker compose build --no-cache
docker compose up
```

* 서버가 실행되면:

```
Uvicorn running on http://0.0.0.0:8000
```

### 4. 종료

```bash
docker compose down
```

---

## 🌐 API 엔드포인트

### 1. 헬스체크

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

### 2. 프레임 개별 전송 (슬라이딩 윈도우)

* **POST** `/predict/frame`
* **Request Body (JSON)**:

```json
{
  "session_id": "user-or-device-uuid",
  "keypoints": [0.12, 0.03, ...]  // 길이 194
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

### 3. NPY 파일 업로드 예측

* **POST** `/predict/npy`
* **Form Data**:

  * file: `.npy` 파일
* **Response**:

```json
{ "label": "지시1#", "confidence": 0.95 }
```

### 4. 세션 초기화

* **DELETE** `/predict/session/{sid}`

### 5. 오래된 세션 정리

* **DELETE** `/predict/sessions/cleanup`

---

## 🧪 API 테스트 예제

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

## 📌 주의사항

* **WINDOW**, **FEATURES**, **CONF\_THRESHOLD** 값은 서버와 클라이언트 모두 동일해야 합니다.
* `session_id`는 각 사용자/기기별로 고유해야 합니다.
* `.npy` 데이터는 `(frames, features)` 형태여야 하며, features는 194로 고정됩니다.
* 로컬이 아닌 외부에서 접속하려면, 서버 IP 또는 도메인을 사용하고 포트를 개방해야 합니다.
* Docker 컨테이너 실행 시 모델 파일(`.h5`, `.json`)이 `/model` 경로에 존재해야 하며, 변경 시 재빌드 필요

---

## 📜 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.