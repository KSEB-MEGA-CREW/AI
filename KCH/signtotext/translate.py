"""
FastAPI backend (Live predict 개선판)
- MediaPipe Holistic로 JPEG 프레임 시퀀스 -> (T=10, D=194) 특징 추출
- 부족한 프레임은 "EAGER_PAD_MODE=dup" 시 마지막 프레임 복제로 채워 즉시 예측 시작
- 중앙 3:4 크롭 + 최대 640 리사이즈로 처리 속도 개선
- max-abs 정규화 후 Keras 모델로 예측
- 응답에 top3, p1/p2/ratio 포함(프런트의 느슨 판정에 사용하세요)

실행:
  uvicorn translate:app --reload --port 8000

주요 환경변수(.env 가능):
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-4.1-mini
  GESTURE_MODEL_PATH=C:\\models\\v5_cnn\\gesture_model.h5
  LABEL_MAP_PATH=C:\\models\\v5_cnn\\label_map.json
  REQUIRED_FRAMES=10
  EXPECTED_LEN=194
  # 라이브 파이프라인 관련
  CONFIDENCE_THRESHOLD=0.96      # 보수적 라벨(label) 판단(프런트는 느슨 로직을 top3로 수행 권장)
  MIN_VALID_FRAMES=6
  MAX_PADDING_RATIO=0.5
  ACCEPT_PARTIAL_SEQ=1           # 1이면 10장 미만이라도 즉시 예측 (부족분 패딩)
  EAGER_PAD_MODE=dup             # dup|zero (부족분을 마지막 프레임 복제 or 0)
  PREPROC_ENABLE=1               # 1이면 서버도 중앙 3:4 크롭+리사이즈
  MAX_SIDE=640                   # 서버 리사이즈 최대 변
"""

import os
import io
import json
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# MediaPipe & OpenCV
import cv2
import mediapipe as mp
from threading import Lock

# -----------------------------
# Load env & constants
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

MODEL_PATH = os.getenv("GESTURE_MODEL_PATH", r"C:\\models\\v5_cnn\\gesture_model.h5")
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", r"C:\\models\\v5_cnn\\label_map.json")

REQUIRED_FRAMES = int(os.getenv("REQUIRED_FRAMES", 10))
EXPECTED_LEN    = int(os.getenv("EXPECTED_LEN", 194))

# Live predict 하이퍼파라미터
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.96))
MAX_PADDING_RATIO    = float(os.getenv("MAX_PADDING_RATIO", 0.5))
MIN_VALID_FRAMES     = int(os.getenv("MIN_VALID_FRAMES", 6))
POSE_SKIP_INDEXES    = set(range(17, 33))  # 하체 제외 (17..32)

# 빠른 응답 위한 옵션
ACCEPT_PARTIAL_SEQ = os.getenv("ACCEPT_PARTIAL_SEQ", "1") == "1"  # 10장 미만 허용
EAGER_PAD_MODE     = os.getenv("EAGER_PAD_MODE", "dup").lower()   # 'dup' or 'zero'

PREPROC_ENABLE     = os.getenv("PREPROC_ENABLE", "1") == "1"
MAX_SIDE           = int(os.getenv("MAX_SIDE", "640"))
TARGET_RATIO       = 3.0 / 4.0

# Gloss vocabulary (14개)
GLOSS_VOCAB = [
    "좋다1", "지시1#", "돕다1", "무엇1", "지시2", "때2", "오늘1",
    "일하다1", "재미1", "필요1", "회사1", "요리1", "괜찮다1", "잘하다2"
]

# -----------------------------
# Safe model loader (AdamW 등 호환)
# -----------------------------
def load_model_safely(path: str):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e1:
        try:
            from tensorflow.keras.optimizers import legacy as legacy_optim
            return tf.keras.models.load_model(
                path, compile=False,
                custom_objects={
                    "Adam": legacy_optim.Adam,
                    "AdamW": getattr(legacy_optim, "AdamW", legacy_optim.Adam),
                },
            )
        except Exception as e2:
            try:
                import keras
                return keras.saving.load_model(path, compile=False, safe_mode=False)
            except Exception as e3:
                raise RuntimeError(f"Model load failed:\n1) {e1}\n2) {e2}\n3) {e3}")

# -----------------------------
# Model & labels
# -----------------------------
_model = None
_label_list: List[str] = []

if os.path.exists(MODEL_PATH):
    try:
        _model = load_model_safely(MODEL_PATH)
        print(f"[INFO] Model loaded: {MODEL_PATH}")
    except Exception as e:
        print("[ERROR] Could not load model:", e)
else:
    print("[WARN] Model path not found:", MODEL_PATH)

if os.path.exists(LABEL_MAP_PATH):
    try:
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            _label_list = json.load(f)
        print(f"[INFO] Label map loaded: {len(_label_list)} labels")
    except Exception as e:
        print("[WARN] Failed to load label_map:", e)
else:
    print("[WARN] Label map not found:", LABEL_MAP_PATH)

# -----------------------------
# OpenAI client (lazy)
# -----------------------------
_openai_client = None
def _get_openai():
    global _openai_client
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Configure your .env or env vars.")
    if _openai_client is None:
        from openai import OpenAI  # type: ignore
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Sign Live Predict API (Fast/Loose-ready)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Schemas
# -----------------------------
class FeaturesPayload(BaseModel):
    features: List[List[float]] = Field(..., description="[[frame][feature]] shape ≈ [REQUIRED_FRAMES, EXPECTED_LEN]")
    topk: int = 3

class WordsPayload(BaseModel):
    words: List[str] = Field(..., description="Predicted tokens/words")
    style: Optional[str] = Field(None, description="예: '존댓말', '간결하게', '반말'")

class PredictAndTranslatePayload(FeaturesPayload):
    style: Optional[str] = None

# -----------------------------
# Utils: training-shape alignment
# -----------------------------
def _pad_or_trim_frames(arr: np.ndarray) -> np.ndarray:
    """Return shape (REQUIRED_FRAMES, EXPECTED_LEN)."""
    if arr.ndim != 2:
        raise ValueError(f"features must be 2D, got {arr.shape}")
    # fix cols
    if arr.shape[1] > EXPECTED_LEN:
        arr = arr[:, :EXPECTED_LEN]
    elif arr.shape[1] < EXPECTED_LEN:
        pad_c = EXPECTED_LEN - arr.shape[1]
        arr = np.hstack([arr, np.zeros((arr.shape[0], pad_c), dtype=arr.dtype)])
    # fix rows
    if arr.shape[0] > REQUIRED_FRAMES:
        arr = arr[:REQUIRED_FRAMES]
    elif arr.shape[0] < REQUIRED_FRAMES:
        pad_r = REQUIRED_FRAMES - arr.shape[0]
        arr = np.vstack([arr, np.zeros((pad_r, EXPECTED_LEN), dtype=arr.dtype)])
    return arr.astype(np.float32)

def predict_topk_from_features(features: List[List[float]], topk: int = 3) -> Dict[str, Any]:
    if _model is None or not _label_list:
        raise RuntimeError("Model/label map not loaded. Check paths or env.")
    arr = np.asarray(features, dtype=np.float32)
    arr = _pad_or_trim_frames(arr)
    probs = _model.predict(arr[None, ...], verbose=0)[0]
    k = max(1, int(topk))
    top_idx = np.argsort(-probs)[:k]
    result = [
        {
            "label": _label_list[int(i)] if int(i) < len(_label_list) else str(int(i)),
            "index": int(i),
            "prob": float(probs[int(i)]),
        }
        for i in top_idx
    ]
    return {"top": result}

# -----------------------------
# MediaPipe (전역 인스턴스 재사용 + 락)
# -----------------------------
mp_holistic = mp.solutions.holistic
_holistic = mp_holistic.Holistic(
    static_image_mode=False,     # 스트리밍/트래킹 모드
    model_complexity=1,
    refine_face_landmarks=False,
    min_detection_confidence=0.6,  # 0.7 -> 0.6 (초기 프레임 인식률↑)
    min_tracking_confidence=0.6,
)
_holistic_lock = Lock()

def _center_crop_resize_3x4_bgr(bgr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """중앙 3:4 크롭 후 긴 변 기준 리사이즈."""
    h, w = bgr.shape[:2]
    src_ratio = w / h
    if src_ratio > TARGET_RATIO:
        new_w = int(h * TARGET_RATIO)
        x0 = (w - new_w) // 2
        bgr = bgr[:, x0:x0 + new_w]
    else:
        new_h = int(w / TARGET_RATIO)
        y0 = (h - new_h) // 2
        bgr = bgr[y0:y0 + new_h, :]
    ch, cw = bgr.shape[:2]
    scale = (max_side / max(ch, cw)) if max(ch, cw) > max_side else 1.0
    if scale != 1.0:
        new_w = int(round(cw * scale))
        new_h = int(round(ch * scale))
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return bgr

def _extract_landmarks(landmarks, dims=3, skip: Optional[set] = None) -> List[float]:
    out: List[float] = []
    if landmarks:
        for i, lm in enumerate(landmarks.landmark):
            if skip and i in skip:
                continue
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, "visibility", 0.0))
            out.extend(coords)
    return out

def _frame_to_194_features_bgr(bgr: np.ndarray) -> Tuple[List[float], bool]:
    """
    1프레임(BGR) -> 길이 194 벡터, 그리고 '유효프레임 여부' 반환
    - hands가 둘다 없으면 invalid
    - landmark 없는 부위는 0으로 패딩
    - zero-ratio > MAX_PADDING_RATIO 이면 invalid
    """
    if PREPROC_ENABLE:
        bgr = _center_crop_resize_3x4_bgr(bgr, MAX_SIDE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with _holistic_lock:
        results = _holistic.process(rgb)

    if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
        return [0.0] * EXPECTED_LEN, False

    lh = _extract_landmarks(results.left_hand_landmarks, dims=3)
    rh = _extract_landmarks(results.right_hand_landmarks, dims=3)
    pose = _extract_landmarks(results.pose_landmarks, dims=4, skip=POSE_SKIP_INDEXES)
    key = lh + rh + pose

    if len(key) < EXPECTED_LEN:
        key += [0.0] * (EXPECTED_LEN - len(key))
    elif len(key) > EXPECTED_LEN:
        key = key[:EXPECTED_LEN]

    zero_ratio = key.count(0.0) / EXPECTED_LEN
    is_valid = zero_ratio <= MAX_PADDING_RATIO
    return key, is_valid

def _decode_jpeg_to_bgr(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    if not jpeg_bytes:
        return None
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img

def images_to_features_bytes(frames_bytes: List[bytes], required_frames: int, expected_len: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    (required_frames x expected_len) 생성:
      - 각 프레임: LH(21x3) + RH(21x3) + POSE(33-하체 17~32, dims=4) = 194
      - hands 둘다 없음 -> invalid
      - zero-ratio > MAX_PADDING_RATIO -> invalid
      - MIN_VALID_FRAMES 미만이면 invalid
      - max-abs 정규화 (max_abs==0 이면 invalid)
    """
    features: List[List[float]] = []
    valid_count = 0
    for jb in frames_bytes[:required_frames]:
        bgr = _decode_jpeg_to_bgr(jb)
        if bgr is None:
            vec = [0.0] * expected_len
            is_valid = False
        else:
            vec, is_valid = _frame_to_194_features_bgr(bgr)
        features.append(vec)
        if is_valid:
            valid_count += 1

    while len(features) < required_frames:
        features.append([0.0] * expected_len)

    seq = np.array(features, dtype=np.float32)  # (T, 194)

    meta = {
        "valid_frames": int(valid_count),
        "required_frames": int(required_frames),
        "zero_padding_ratio_seq": float(np.mean((seq == 0.0).astype(np.float32))),
    }

    if valid_count < MIN_VALID_FRAMES:
        meta["invalid_reason"] = f"valid_frames<{MIN_VALID_FRAMES}"
        return seq, meta

    max_abs = float(np.max(np.abs(seq)))
    if max_abs <= 0:
        meta["invalid_reason"] = "max_abs<=0"
        return seq, meta

    seq /= max_abs
    meta["normalized"] = True
    meta["max_abs"] = max_abs
    return seq, meta

# -----------------------------
# Words → Sentence (OpenAI)
# -----------------------------
def words_to_sentence(words: List[str], style: Optional[str] = None) -> str:
    tokens = [w for w in words if w and w in GLOSS_VOCAB and w.lower() != "none"]
    if not tokens:
        return "(빈 입력)"

    style_hint = (style or "존댓말").strip()
    if "존댓" in style_hint:
        default_you = "당신"; default_me = "저"; what_word = "무엇"
    else:
        default_you = "너"; default_me = "나"; what_word = "뭐"

    system_msg = (
        "당신은 수어 글로스(gloss) 시퀀스를 한국어 한 문장으로 자연스럽게 바꾸는 전문가입니다. "
        "입력 글로스는 아래 14개 중에서만 옵니다. 숫자 접미사는 변이 번호이므로 의미 해석에는 영향을 주지 않습니다.\n"
        f"- 허용 글로스: {', '.join(GLOSS_VOCAB)}\n"
        "- 지시1#: 상대방/사물 지시(2인칭 혹은 지시대명사). 맥락이 없으면 2인칭으로 해석.\n"
        "- 지시2 : 화자(1인칭) 지시.\n"
        "규칙:\n"
        "1) 글로스 순서는 크게 바꾸지 말되, 한국어 어순에 맞게 최소한으로 재배열.\n"
        "2) 의미 왜곡 금지. 새 명사를 임의로 추가하지 말 것. 다만 조사/어미/필수 문법 기능어는 자연스럽게 보충 가능.\n"
        "3) 맞춤법/띄어쓰기/어미를 자연스럽게 보정.\n"
        "4) 출력은 문장 1개만.\n"
    )
    user_msg = (
        "입력 글로스들을 자연스러운 한국어 한 문장으로 바꿔주세요.\n"
        f"- 스타일: {style_hint}\n"
        f"- 디폴트 대명사 지침(맥락 없음 가정): 지시1#→\"{default_you}\", 지시2→\"{default_me}\", 무엇1→\"{what_word}\"\n"
        "- 필요 시 존칭/반말은 스타일에 맞춰 적용하세요.\n"
        f"- 입력 글로스: {', '.join(tokens)}\n"
        "→ 결과: "
    )

    client = _get_openai()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.2,
        max_tokens=120,
    )
    text = resp.choices[0].message.content.strip()
    if text and text[-1] not in "?!。”.":  # 마침표 보정
        text += "."
    return text

# -----------------------------
# Routes
# -----------------------------
@app.get("/api/health")
async def health():
    return {"ok": True, "model_loaded": bool(_model is not None), "labels": len(_label_list)}

@app.post("/api/predict-words-from-features")
async def predict_words_from_features(payload: FeaturesPayload):
    try:
        out = predict_topk_from_features(payload.features, payload.topk)
        return {"ok": True, **out}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/translate-words")
async def translate_words(payload: WordsPayload):
    try:
        sentence = words_to_sentence(payload.words, payload.style)
        return {"ok": True, "sentence": sentence, "tokens": payload.words}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict-and-translate")
async def predict_and_translate(payload: PredictAndTranslatePayload):
    try:
        pred = predict_topk_from_features(payload.features, topk=1)
        top_label = pred["top"][0]["label"]
        sentence = words_to_sentence([top_label], payload.style)
        return {"ok": True, "top": pred["top"], "sentence": sentence}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 업로드(데이터셋 저장)
@app.post("/api/upload-sequence")
async def upload_sequence(
    frames: List[UploadFile] = File(...),
    label: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),
):
    save_dir = os.path.join(os.getcwd(), "webcam_uploads")
    os.makedirs(save_dir, exist_ok=True)
    sub = f"{label or 'unlabeled'}"
    if timestamp:
        safe_ts = timestamp.replace(":", "-")
        sub += f"_{safe_ts}"
    subdir = os.path.join(save_dir, sub)
    os.makedirs(subdir, exist_ok=True)

    saved = []
    for i, f in enumerate(frames):
        dst = os.path.join(subdir, f"frame_{i:03d}.jpg")
        with open(dst, "wb") as out:
            out.write(await f.read())
        saved.append(dst)
    return {"ok": True, "saved": saved, "dir": subdir}

# 핵심: 프레임 시퀀스 -> 단어
@app.post("/api/predict-word-from-frames")
async def predict_word_from_frames(
    frames: List[UploadFile] = File(...),
    mock: Optional[str] = Form(None),  # '1'이면 랜덤 라벨 응답(데모용)
):
    try:
        have_model = (_model is not None and len(_label_list) > 0)

        if (not have_model) and (mock == "1"):
            import random
            return {"ok": True, "label": random.choice(_label_list or ["none"]), "mock": True}

        # bytes로 읽기
        fb = [await f.read() for f in frames]

        # ✅ EAGER: 10장 미만이면 마지막 프레임 복제로 10장 맞춤 (초기 예측 체감속도↑)
        if len(fb) < REQUIRED_FRAMES and ACCEPT_PARTIAL_SEQ:
            if len(fb) == 0:
                # 정말 아무 것도 없는 경우는 0패딩
                fb = [b""] * REQUIRED_FRAMES
            else:
                pad_src = fb[-1] if EAGER_PAD_MODE == "dup" else b""
                fb = fb + [pad_src] * (REQUIRED_FRAMES - len(fb))
        else:
            fb = fb[:REQUIRED_FRAMES]
            if len(fb) < REQUIRED_FRAMES:  # (ACCEPT_PARTIAL_SEQ=0일 때를 대비)
                fb = fb + [b""] * (REQUIRED_FRAMES - len(fb))

        feats, meta = images_to_features_bytes(fb, REQUIRED_FRAMES, EXPECTED_LEN)  # (T,194)

        if ("invalid_reason" in meta) or np.max(np.abs(feats)) <= 0:
            return {
                "ok": True, "label": "none", "confidence": 0.0,
                "raw_label": "none", "top3": [], "meta": meta, "mock": False
            }

        if _model is None or not _label_list:
            raise RuntimeError("Model/labels not loaded on server.")

        x = feats[None, ...]  # (1,T,194)
        probs = _model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(np.max(probs))
        raw_label = _label_list[idx] if idx < len(_label_list) else str(idx)
        label = raw_label if conf >= CONFIDENCE_THRESHOLD else "none"

        # top3 + p1/p2/ratio 제공(프런트 느슨 판정용)
        top_idx = np.argsort(-probs)[:3]
        top3 = [
            {"index": int(i),
             "label": _label_list[int(i)] if int(i) < len(_label_list) else str(int(i)),
             "prob": float(probs[int(i)])}
            for i in top_idx
        ]
        p1 = float(probs[top_idx[0]]) if len(top_idx) > 0 else 0.0
        p2 = float(probs[top_idx[1]]) if len(top_idx) > 1 else 0.0
        ratio = float(p1 / max(1e-6, p1 + p2))

        return {
            "ok": True,
            "label": label,
            "confidence": conf,
            "raw_label": raw_label,
            "threshold": CONFIDENCE_THRESHOLD,
            "top3": top3,
            "p1": p1, "p2": p2, "ratio": ratio,
            "meta": meta,
            "mock": False,
        }

    except Exception as e:
        return {"ok": False, "detail": f"{type(e).__name__}: {e}"}

# (optional) main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("translate:app", host="0.0.0.0", port=8000, reload=True)