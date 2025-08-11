import os
import numpy as np
import json
from tensorflow.keras.models import load_model

# ===== 설정 =====
MODEL_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\models\test2\gesture_model.h5"
LABEL_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\models\test2\label_map.json"
DATA_DIR = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\output_npy\cleaned_npy\좋다1"  # 예측할 npy 폴더
REQUIRED_FRAMES = 12
EXPECTED_LEN = 194
MIN_VALID_FRAMES = 5
MAX_PADDING_RATIO = 0.44

# ===== 모델 및 라벨 로딩 =====
model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_list = json.load(f)

# ===== 전처리 함수 =====
def preprocess_npy(npy_path):
    seq = np.load(npy_path)
    if seq.shape[0] < MIN_VALID_FRAMES:
        return None

    if seq.shape[0] < REQUIRED_FRAMES:
        pad = np.zeros((REQUIRED_FRAMES - seq.shape[0], EXPECTED_LEN))
        seq = np.vstack([seq, pad])
    else:
        seq = seq[:REQUIRED_FRAMES]

    if np.sum(seq == 0) / seq.size > MAX_PADDING_RATIO:
        return None

    max_abs = np.max(np.abs(seq))
    if max_abs > 0:
        seq = seq / max_abs

    return np.expand_dims(seq, axis=0)

# ===== 예측 반복 =====
print(f"\n📂 예측할 폴더: {DATA_DIR}")
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
if not files:
    print("❌ 예측할 npy 파일이 없습니다.")
else:
    print(f"✅ 총 {len(files)}개 파일 예측 시작:\n")

    for i, fname in enumerate(files, 1):
        path = os.path.join(DATA_DIR, fname)
        input_seq = preprocess_npy(path)

        if input_seq is None:
            print(f"[{i:02d}] {fname}: ❌ 유효하지 않은 입력")
            continue

        pred_probs = model.predict(input_seq, verbose=0)[0]
        pred_idx = np.argmax(pred_probs)
        pred_label = label_list[pred_idx]
        confidence = pred_probs[pred_idx]

        # 파일명 기준 실제 라벨 추출 (예: 지시1#_0001.npy → 지시1#)
        true_label = os.path.basename(DATA_DIR)

        if pred_label != true_label:
            print(f"[{i:02d}] {fname:35s} → ❌ {pred_label} (정답: {true_label}) → 삭제됨")
            os.remove(path)
        else:
            print(f"[{i:02d}] {fname:35s} → ✅ {pred_label} | 정확도: {confidence:.4f}")