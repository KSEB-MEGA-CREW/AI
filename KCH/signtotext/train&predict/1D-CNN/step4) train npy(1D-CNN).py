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

# 윈도우 기준, 'Malgun Gothic' 폰트 지정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 경로 설정 (절대경로)
DATA_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\output_npy"
required_frames = 35
expected_len = 194
random.seed(42)

# 1. [정면 데이터]만 라벨별로 분류
label_files = defaultdict(list)
for fname in os.listdir(DATA_PATH):
    # ▶ '정면' 데이터: 언더바 1개만 있는 파일만 사용
    if fname.endswith(".npy") and fname.count("_") == 1:
        label = fname.split("_")[1].split('.')[0]
        label_files[label].append(fname)

# 2. 데이터 개수 많은 라벨 Top-5 추출
sorted_labels = sorted(label_files.items(), key=lambda x: len(x[1]), reverse=True)
selected_labels = [label for label, files in sorted_labels[:5]]
print(f"선택된 라벨(Top-5): {selected_labels}")

# 3. 각 라벨별 파일 개수 확인 & 최소 샘플 수 산출
min_count = min(len(label_files[label]) for label in selected_labels)
print(f"선택된 라벨별 파일 수: {[len(label_files[label]) for label in selected_labels]}")
print(f"모든 라벨에서 사용될 샘플 수(최소값): {min_count}")

# 4. 최소 샘플 수만큼 랜덤 추출해 데이터셋 구성
sequences, labels = [], []
label_dict = {label: i for i, label in enumerate(selected_labels)}

for label in selected_labels:
    files = label_files[label]
    chosen_files = random.sample(files, min_count)
    label_num = label_dict[label]
    for file in chosen_files:
        path = os.path.join(DATA_PATH, file)
        sequence = np.load(path)
        # 시계열 길이 맞추기 (패딩/자르기)
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

# 5. 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6. 1D-CNN (무거운 구조)
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

# 7. 학습
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

# 8. 저장
os.makedirs("models", exist_ok=True)
model.save("models/gesture_model.h5")
label_list = [label for label, idx in sorted(label_dict.items(), key=lambda x: x[1])]
with open("models/label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_list, f, ensure_ascii=False)

# 9. 시각화
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