from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from tensorflow.keras.models import load_model
import numpy as np
import json
from io import BytesIO
from collections import deque
import time
import logging

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# --- CORS (프론트 연동 시 편의) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 도메인 제한으로 바꾸세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 모델 & 라벨 경로 =====
MODEL_PATH = "../model/frame_to_gloss_v0.h5"
LABEL_PATH = "../model/frame_to_gloss_v0.json"

# ===== 서버 설정 =====
WINDOW = 10        # 슬라이딩 윈도우 프레임 수
FEATURES = 194     # 프레임당 피처 수 (프로젝트 기준)
CONF_THRESHOLD = 0.4
SESSION_TTL = 300  # 세션 비활성 정리 기준(초) - 5분

# ===== 모델/라벨 로드 =====
model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_list = json.load(f)

# ===== 세션 상태(메모리) =====
# session_id -> deque, last_seen
buffers: Dict[str, deque] = {}
last_seen: Dict[str, float] = {}

# ===== 헬스 체크 =====
@app.get("/health")
def health():
    return {
        "status": "ok",
        "window": WINDOW,
        "features": FEATURES,
        "sessions": len(buffers)
    }

# ===== 배치(.npy) 업로드 예측 (옵션) =====
@app.post("/predict/npy")
async def predict_npy(file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=415, detail="Only .npy files are supported.")

    try:
        raw = await file.read()
        arr = np.load(BytesIO(raw), allow_pickle=False)

        if arr.ndim == 1:
            raise HTTPException(status_code=422, detail=f"Invalid shape {arr.shape}. Expected (frames, features).")
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.shape[1] != FEATURES:
            logger.warning(f"Unexpected features: got {arr.shape[1]}, expected {FEATURES}")

        x = np.expand_dims(arr, axis=0)  # (1, frames, features)
        probs = model.predict(x, verbose=0)
        idx = int(np.argmax(probs, axis=-1)[0])
        confidence = float(np.max(probs, axis=-1)[0])
        label = label_list[idx] if confidence >= CONF_THRESHOLD else "None"
        return {"label": label, "confidence": confidence}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"/predict/npy failed: {e}")
        raise HTTPException(status_code=400, detail=f"Error: {e}")

# ===== 프레임 개별 전송용 스키마 =====
class FrameIn(BaseModel):
    session_id: str
    keypoints: List[float]  # 길이 FEATURES

# ===== 프레임 개별 전송: 세션별 버퍼에 쌓아 10프레임 채워지면 예측 =====
@app.post("/predict/frame")
async def predict_frame(payload: FrameIn = Body(...)):
    sid = payload.session_id
    kp = payload.keypoints

    if not isinstance(kp, list) or len(kp) != FEATURES:
        raise HTTPException(status_code=422, detail=f"keypoints must be length {FEATURES}")

    # 세션 버퍼 준비
    if sid not in buffers:
        buffers[sid] = deque(maxlen=WINDOW)
    buffers[sid].append(kp)
    last_seen[sid] = time.time()

    # 아직 윈도우가 안 찼으면 상태만 반환
    collected = len(buffers[sid])
    if collected < WINDOW:
        return {"status": "collecting", "collected": collected, "window": WINDOW}

    # 윈도우 채워졌으면 예측
    try:
        arr = np.array(buffers[sid], dtype=np.float32)  # (WINDOW, FEATURES)
        x = np.expand_dims(arr, axis=0)                 # (1, WINDOW, FEATURES)
        probs = model.predict(x, verbose=0)
        idx = int(np.argmax(probs, axis=-1)[0])
        confidence = float(np.max(probs, axis=-1)[0])
        label = label_list[idx] if confidence >= CONF_THRESHOLD else "None"
        return {"label": label, "confidence": confidence, "window": WINDOW}
    except Exception as e:
        logger.exception(f"/predict/frame failed: {e}")
        raise HTTPException(status_code=400, detail=f"Error: {e}")

# ===== 세션 초기화/정리 API =====
@app.delete("/predict/session/{sid}")
def clear_session(sid: str):
    buffers.pop(sid, None)
    last_seen.pop(sid, None)
    return {"cleared": sid}

@app.delete("/predict/sessions/cleanup")
def cleanup_sessions():
    now = time.time()
    removed = []
    for sid, ts in list(last_seen.items()):
        if now - ts > SESSION_TTL:
            removed.append(sid)
            buffers.pop(sid, None)
            last_seen.pop(sid, None)
    return {"removed": removed}