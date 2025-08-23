# Dockerfile (API)
FROM python:3.10-slim

# 시스템 패키지 (mediapipe/opencv가 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 파이썬 의존성 설치
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 앱 코드
COPY app /app/app

# 모델은 컨테이너에 copy 할 수도 있고, compose로 마운트할 수도 있음
# 여기서는 폴더만 만들어 두고, compose에서 마운트(ro) 합니다.
RUN mkdir -p /app/model

ENV MODEL_DIR=/app/model
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]