import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import defaultdict
import json
import random
import matplotlib
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows 기준)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# === 경로 및 하이퍼파라미터 ===
DATA_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\output_npy"
REQUIRED_FRAMES = 35
EXPECTED_LEN = 194

# === 사용자 입력값 받기 ===
try:
    TOP_N = int(input("학습할 라벨 개수(예: 30): ").strip())
except Exception:
    TOP_N = 30
print(f"학습할 라벨 개수: {TOP_N}")

try:
    MIN_SAMPLES = int(input("라벨별 최소 데이터 개수 이상만 포함(예: 30): ").strip())
except Exception:
    MIN_SAMPLES = 30
print(f"라벨별 최소 데이터 개수: {MIN_SAMPLES}")

random.seed(42)

# === 1. '정면' 파일만 라벨별로 분류 (예: [임의문자열]_[라벨]_C.npy) ===
label_files = defaultdict(list)
for fname in os.listdir(DATA_PATH):
    if fname.endswith("_C.npy"):
        label = fname.split("_")[1]
        label_files[label].append(fname)

print("\n[라벨별 npy 파일 개수(내림차순)]")
label_count_list = sorted(label_files.items(), key=lambda x: len(x[1]), reverse=True)
for label, files in label_count_list:
    print(f"{label:15s}: {len(files)}개")
print(f"\n총 라벨 수: {len(label_files)}개")

# === 2. MIN_SAMPLES 이상 데이터 가진 라벨만 추출 → TOP_N개만 선택 ===
eligible_labels = [label for label, files in label_files.items() if len(files) >= MIN_SAMPLES]
print(f"\n[{MIN_SAMPLES}개 이상 npy 가진 라벨 수]: {len(eligible_labels)}개")

sorted_labels = sorted([(label, label_files[label]) for label in eligible_labels],
                      key=lambda x: len(x[1]), reverse=True)

print(f"\n[조건 만족 라벨 TOP-{TOP_N} 미리보기]")
for label, files in sorted_labels[:TOP_N]:
    print(f"{label:15s}: {len(files)}개")

if len(eligible_labels) < TOP_N:
    raise ValueError(f"\nMIN_SAMPLES={MIN_SAMPLES} 기준을 만족하는 라벨이 {len(eligible_labels)}개뿐입니다. 값을 조정하세요.")

selected_labels = [label for label, files in sorted_labels[:TOP_N]]
print(f"\n[최종 학습 라벨 목록 ({TOP_N}개)]")
print(selected_labels)

# === 3. 각 라벨별 MIN_SAMPLES만큼만 랜덤 추출하여 데이터셋 구성 ===
sequences, labels = [], []
label_dict = {label: i for i, label in enumerate(selected_labels)}
for label in selected_labels:
    files = label_files[label]
    chosen_files = random.sample(files, MIN_SAMPLES)
    label_num = label_dict[label]
    for file in chosen_files:
        path = os.path.join(DATA_PATH, file)
        sequence = np.load(path)
        # 프레임 수 맞춤(패딩/자르기)
        if sequence.shape[0] < REQUIRED_FRAMES:
            pad = np.zeros((REQUIRED_FRAMES - sequence.shape[0], EXPECTED_LEN))
            sequence_fixed = np.vstack([sequence, pad])
        else:
            sequence_fixed = sequence[:REQUIRED_FRAMES]
        # 정규화
        max_abs = np.max(np.abs(sequence_fixed))
        if max_abs > 0:
            sequence_fixed = sequence_fixed / max_abs
        sequences.append(sequence_fixed)
        labels.append(label_num)

X = np.array(sequences)
y = to_categorical(labels)
print(f"\n[최종 데이터 shape] X={X.shape}, y={y.shape}, 라벨={len(label_dict)}개")

# === 4. 학습/검증 분리 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# === 5. 1D-CNN 모델 정의 ===
model = Sequential([
    Conv1D(128, 7, activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(256, 5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(256, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 6. 콜백 및 학습 ===
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === 7. 모델 및 라벨맵 저장 ===
os.makedirs("models", exist_ok=True)
model.save("models/gesture_model.h5")
label_list = [label for label, idx in sorted(label_dict.items(), key=lambda x: x[1])]
with open("models/label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_list, f, ensure_ascii=False)

# === 8. 학습 곡선 시각화 ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='훈련 정확도', marker='o')
plt.plot(history.history['val_accuracy'], label='검증 정확도', marker='x')
plt.title("정확도 변화")
plt.xlabel("Epoch")
plt.ylabel("정확도")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='훈련 손실', marker='o')
plt.plot(history.history['val_loss'], label='검증 손실', marker='x')
plt.title("손실 변화")
plt.xlabel("Epoch")
plt.ylabel("손실")
plt.legend()

plt.tight_layout()
plt.show()