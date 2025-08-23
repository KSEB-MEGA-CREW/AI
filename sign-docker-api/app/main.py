# app/main.py
import os
import io
import json
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Image / MediaPipe
from PIL import Image
import mediapipe as mp

# OpenAI (선택)
try:
    import openai  # 구버전 호환
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# -----------------------
# 전역 설정
# -----------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "label_map.json")

EXPECTED_KEYPOINTS = 194
REQUIRED_FRAMES = 10
POSE_SKIP_INDEXES = set(range(17, 33))  # 하체 제외

# TensorFlow 로그 줄이기
tf.get_logger().setLevel("ERROR")

# -----------------------
# FastAPI
# -----------------------
app = FastAPI(title="Sign2Korean API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영에서는 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# 모델 / 라벨 로드
# -----------------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")
if not os.path.exists(LABEL_PATH):
    raise RuntimeError(f"Label map not found: {LABEL_PATH}")

model = load_model(MODEL_PATH, compile=False)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_list = json.load(f)
idx2label = {i: s for i, s in enumerate(label_list)}

print(f"[INFO] Model loaded: {MODEL_PATH}")
print(f"[INFO] Label map loaded: {len(label_list)} labels")

# -----------------------
# 유틸 함수
# -----------------------
mp_holistic = mp.solutions.holistic

def pil_jpeg_to_rgb_array(file_bytes: bytes) -> np.ndarray:
    """JPEG bytes -> RGB np.array"""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)

def extract_landmarks(landmarks, dims=3, skip=None):
    """MediaPipe landmark -> flat list"""
    result = []
    if landmarks:
        for i, lm in enumerate(landmarks.landmark):
            if skip and i in skip:
                continue
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, "visibility", 0.0))
            result.extend(coords)
    return result

def frames_to_tensor(frames_jpegs: List[bytes]) -> np.ndarray:
    """
    JPEG 목록(최대 최근 10장)을 MediaPipe로 처리 → (10,194) float32 텐서 생성
    - 프레임 부족 시 0패딩
    - 프레임 초과 시 최근 10장만 사용
    - 입력 정규화: max-abs 로 나눔
    """
    # 최근 10개만
    frames_jpegs = frames_jpegs[-REQUIRED_FRAMES:]

    seq_rows = []
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.6,
    ) as holistic:
        for b in frames_jpegs:
            rgb = pil_jpeg_to_rgb_array(b)
            results = holistic.process(rgb)

            # 손이 모두 없으면 zeros (클라이언트와 동일 철학)
            if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
                seq_rows.append(np.zeros((EXPECTED_KEYPOINTS,), dtype=np.float32))
                continue

            lh = extract_landmarks(results.left_hand_landmarks, dims=3)
            rh = extract_landmarks(results.right_hand_landmarks, dims=3)
            pose = extract_landmarks(results.pose_landmarks, dims=4, skip=POSE_SKIP_INDEXES)
            keypoints = lh + rh + pose

            # 길이 맞추기
            if len(keypoints) < EXPECTED_KEYPOINTS:
                keypoints += [0.0] * (EXPECTED_KEYPOINTS - len(keypoints))
            elif len(keypoints) > EXPECTED_KEYPOINTS:
                keypoints = keypoints[:EXPECTED_KEYPOINTS]

            seq_rows.append(np.array(keypoints, dtype=np.float32))

    # 프레임 개수 맞추기 (앞쪽 0패딩)
    if len(seq_rows) < REQUIRED_FRAMES:
        pad_count = REQUIRED_FRAMES - len(seq_rows)
        pad = [np.zeros((EXPECTED_KEYPOINTS,), dtype=np.float32)] * pad_count
        seq_rows = pad + seq_rows

    seq = np.stack(seq_rows, axis=0).astype(np.float32)  # (10,194)

    # 정규화 (max-abs)
    max_abs = float(np.max(np.abs(seq))) if np.max(np.abs(seq)) > 0 else 1.0
    seq = seq / max_abs

    return seq  # (10,194)

def run_model_on_tensor(seq: np.ndarray) -> Dict[str, Any]:
    """(10,194) -> top3 결과 반환"""
    x = np.expand_dims(seq, axis=0)  # (1, 10, 194)
    probs = model.predict(x, verbose=0)[0]  # (num_classes,)
    # 이미 softmax라고 가정. 아니라면 아래 한 줄로 보정 가능
    # probs = tf.nn.softmax(probs).numpy()

    top_idx = np.argsort(probs)[::-1][:3]
    top3 = [{"label": idx2label[i], "prob": float(probs[i])} for i in top_idx]
    raw_label = top3[0]["label"]
    confidence = top3[0]["prob"]
    return {"raw_label": raw_label, "confidence": float(confidence), "top3": top3}

# -----------------------
# 번역(문장 생성)
# -----------------------
class WordsPayload(BaseModel):
    words: List[str] = []
    style: str = "존댓말"

def gloss_to_korean_token(gloss: str) -> str:
    """간단한 gloss → 토큰 변환 (백업용)"""
    m = {
        "지시1#": "너",   # 상대 지시
        "지시2": "나",    # 자기 지시
        "오늘1": "오늘",
        "회사1": "회사",
        "일하다1": "일하다",
        "재미1": "재미있다",
        "필요1": "필요하다",
        "괜찮다1": "괜찮다",
        "돕다1": "돕다",
        "무엇1": "무엇",
        "좋다1": "좋다",
        "요리1": "요리하다",
        "잘하다2": "잘하다",
        "때2": "때"
    }
    return m.get(gloss, gloss)

async def openai_sentence(words: List[str], style: str) -> str:
    """OpenAI 사용 (환경변수 없는 경우 예외)"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not (HAS_OPENAI and api_key):
        raise RuntimeError("OpenAI unavailable")

    openai.api_key = api_key
    # 간단 프롬프트
    prompt = (
        "아래 수어 글로스 단어 목록을 자연스러운 한국어 한 문장으로 변환해 주세요.\n"
        f"- 글로스: {words}\n"
        f"- 톤/스타일: {style}\n"
        "- 지시1#는 '너(상대)'를, 지시2는 '나(화자)'를 의미합니다.\n"
        "- 의미를 보존하되, 조사/어미를 자연스럽게 붙여 문장으로 출력하세요.\n"
        "- 출력은 한 문장만."
    )

    # 구버전 호환 ChatCompletion
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
        max_tokens=60,
    )
    return resp["choices"][0]["message"]["content"].strip()

def rule_based_sentence(words: List[str]) -> str:
    """키 없거나 실패 시 간단 백업 규칙"""
    toks = [gloss_to_korean_token(w) for w in words if w and w.lower() != "none"]
    if not toks:
        return "(인식된 단어 없음)"
    # 아주 단순 조립
    sent = " ".join(toks)
    # 마침표/존댓말 보정
    if not sent.endswith(("다", "요", ".", "!", "?")):
        sent += "입니다."
    return sent

# -----------------------
# 라우팅
# -----------------------
@app.get("/healthz")
async def health() -> Dict[str, Any]:
    return {"ok": True}

@app.post("/api/predict-word-from-frames")
async def predict(frames: List[UploadFile] = File(...)) -> Dict[str, Any]:
    if not frames:
        raise HTTPException(status_code=400, detail="frames required")

    # 바이트 추출
    try:
        jpegs = [await f.read() for f in frames]
    except Exception:
        raise HTTPException(status_code=400, detail="failed to read frames")

    # 텐서 변환
    try:
        seq = frames_to_tensor(jpegs)  # (10,194)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"preprocess error: {type(e).__name__}")

    # 추론
    try:
        out = run_model_on_tensor(seq)
        return {"ok": True, **out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"infer error: {type(e).__name__}")

@app.post("/api/translate-words")
async def translate(payload: WordsPayload) -> Dict[str, Any]:
    words = payload.words or []
    style = payload.style or "존댓말"
    try:
        try:
            sentence = await openai_sentence(words, style)
        except Exception:
            sentence = rule_based_sentence(words)
        return {"ok": True, "sentence": sentence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"translate error: {type(e).__name__}")