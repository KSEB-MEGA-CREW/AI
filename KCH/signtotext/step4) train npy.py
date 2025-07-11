import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
import json

# 🔹 한글 폰트 설정 (그래프용)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 🔹 데이터 로드
DATA_PATH = "output_npy"
sequences = []
labels = []
label_dict = {}
label_idx = 0

expected_len = 194      # 한 프레임의 feature 길이
required_frames = 35    # 시계열 프레임 길이

print("[🔍 데이터 로딩 중...]")
for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        try:
            label = file.split("_")[1]  # gloss_id
        except Exception as e:
            print(f"❌ 파일명 파싱 오류: {file}")
            continue

        path = os.path.join(DATA_PATH, file)
        try:
            sequence = np.load(path)
            # ------- 수정 부분 시작 -------
            # 프레임 수가 부족하면 0으로 패딩 (초과하면 자름)
            seq_len = sequence.shape[0]
            if seq_len < required_frames:
                pad = np.zeros((required_frames - seq_len, expected_len))
                sequence_fixed = np.vstack([sequence, pad])
            else:
                sequence_fixed = sequence[:required_frames]
            # ------- 수정 부분 끝 -------

            # 라벨 인덱스 할당
            if label not in label_dict:
                label_dict[label] = label_idx
                label_idx += 1
            label_num = label_dict[label]

            sequences.append(sequence_fixed)
            labels.append(label_num)

        except Exception as e:
            print(f"❌ 오류: {file} → {e}")

# 🔹 배열로 변환
X = np.array(sequences)
y = to_categorical(labels)

print(f"✅ 데이터 로딩 완료! X shape: {X.shape}, y shape: {y.shape}")
print(f"📄 라벨 매핑: {label_dict}")

# 🔹 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 🔹 모델 구성 (1D-CNN)
model = Sequential([
    Conv1D(128, 7, activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

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

# 🔹 학습
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# 🔹 모델 및 라벨 저장
os.makedirs("models", exist_ok=True)
model.save("models/gesture_model.h5")
print("✅ 모델 저장 완료: models/gesture_model.h5")

# 🔹 라벨맵 저장
sorted_labels = [label for label, idx in sorted(label_dict.items(), key=lambda x: x[1])]
with open("models/label_map.json", "w", encoding="utf-8") as f:
    json.dump(sorted_labels, f, ensure_ascii=False)
print("✅ label_map.json 저장 완료!")

# 🔹 시각화
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