import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, GlobalAveragePooling1D, Dense
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

# 데이터 경로 및 설정
DATA_PATH = "output_npy"
required_frames = 35
expected_len = 194
SAMPLE_COUNT = 306  # 라벨당 사용할 샘플 수
random.seed(42)

# 1. 라벨별 파일 분류
label_files = defaultdict(list)
for fname in os.listdir(DATA_PATH):
    if fname.endswith(".npy"):
        label = fname.split("_")[1].split('.')[0]
        label_files[label].append(fname)

# 2. 충분한 수의 샘플이 있는 라벨만 선택
selected_labels = [label for label, files in label_files.items() if len(files) >= SAMPLE_COUNT]
print(f"{SAMPLE_COUNT}개 이상 라벨({len(selected_labels)}개):", selected_labels)

# 3. 시퀀스 및 라벨 생성
sequences, labels = [], []
label_dict, label_idx = {}, 0
for label in selected_labels:
    files = label_files[label]
    chosen_files = random.sample(files, SAMPLE_COUNT)
    if label not in label_dict:
        label_dict[label] = label_idx
        label_idx += 1
    label_num = label_dict[label]
    for file in chosen_files:
        path = os.path.join(DATA_PATH, file)
        sequence = np.load(path)

        # 프레임 수 맞추기
        if sequence.shape[0] < required_frames:
            pad = np.zeros((required_frames - sequence.shape[0], expected_len))
            sequence_fixed = np.vstack([sequence, pad])
        else:
            sequence_fixed = sequence[:required_frames]

        # 정규화
        max_abs = np.max(np.abs(sequence_fixed))
        if max_abs > 0:
            sequence_fixed = sequence_fixed / max_abs

        sequences.append(sequence_fixed)
        labels.append(label_num)

X = np.array(sequences)
y = to_categorical(labels)
print(f"최종 데이터 shape: X={X.shape}, y={y.shape}, 라벨={len(label_dict)}개")

# 4. 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Conv1D + GAP 모델 정의
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    GlobalAveragePooling1D(),

    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. 학습
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

# 7. 모델 저장
os.makedirs("models", exist_ok=True)
model.save("models/gesture_model_conv1d_gap.h5")
sorted_labels = [label for label, idx in sorted(label_dict.items(), key=lambda x: x[1])]
with open("models/label_map_gap.json", "w", encoding="utf-8") as f:
    json.dump(sorted_labels, f, ensure_ascii=False)

# 8. 학습 결과 시각화
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