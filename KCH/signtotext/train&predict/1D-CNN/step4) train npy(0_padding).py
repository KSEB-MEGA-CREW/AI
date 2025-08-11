import os
import numpy as np
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 🔸 경로 설정
DATA_PATH = r"C:\cleaned_npy"
SAVE_DIR = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\models\final_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔸 전처리 및 모델 설정
REQUIRED_FRAMES = 10
EXPECTED_LEN = 194

# --- 데이터 로딩 (기존 코드와 동일) ---
label_files = defaultdict(list)
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".npy"):
            label = os.path.basename(root)
            label_files[label].append(os.path.join(root, file))

label_count_list = sorted(label_files.items(), key=lambda x: len(x[1]), reverse=True)
print("\n📊 라벨별 npy 개수:")
for i, (label, files) in enumerate(label_count_list, 1):
    print(f"{i:3d}. {label:15s}: {len(files)}개")

# --- [수정] 사용자 입력 및 데이터 선택 로직 개선 ---
try:
    TOP_N = int(input("\n👉 학습할 라벨 개수(예: 30, 'none' 제외): ").strip())
except Exception:
    TOP_N = 30
print(f"✅ 학습할 라벨 개수: {TOP_N}")

try:
    # [수정] 변수명을 SAMPLES_PER_CLASS로 변경하여 의미 명확화
    SAMPLES_PER_CLASS = int(input("👉 라벨별로 사용할 최대 데이터 개수(예: 320): ").strip())
except Exception:
    SAMPLES_PER_CLASS = 320
print(f"✅ 라벨별 최대 데이터 개수: {SAMPLES_PER_CLASS}")

# 'none'을 제외한 라벨들 중에서 빈도수 높은 순으로 선택
selected_labels = [label for label, files in label_count_list if label != "none"][:TOP_N]

# [수정] 'none' 클래스가 존재하면 무조건 학습에 포함
if "none" in label_files:
    selected_labels.append("none")
    print("✅ 'none' 클래스를 학습에 자동으로 추가합니다.")

label_dict = {label: i for i, label in enumerate(selected_labels)}
print(f"\n✅ 최종 학습 라벨 목록 ({len(selected_labels)}개):\n{selected_labels}")


# --- [수정] 데이터 증강 함수 정의 ---
def augment_sequence(sequence, noise_level=0.005):
    """ 데이터에 작은 노이즈를 추가하여 증강합니다. """
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise


# --- 데이터 전처리 (샘플링 방식 및 증강 적용) ---
sequences, labels = [], []
for label in selected_labels:
    all_files = label_files[label]

    # [수정] SAMPLES_PER_CLASS만큼 데이터를 사용하도록 로직 변경
    if len(all_files) > SAMPLES_PER_CLASS:
        files_to_use = random.sample(all_files, SAMPLES_PER_CLASS)
    else:
        files_to_use = all_files

    for file_path in files_to_use:
        seq = np.load(file_path)

        # 프레임 길이 맞추기 (Padding / Truncating)
        if seq.shape[0] < REQUIRED_FRAMES:
            pad = np.zeros((REQUIRED_FRAMES - seq.shape[0], EXPECTED_LEN))
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:REQUIRED_FRAMES]

        # 원본 데이터 추가
        sequences.append(seq)
        labels.append(label_dict[label])

        # [수정] 데이터 증강: 데이터가 부족한 클래스는 2배로 증강 (none 제외)
        if label != "none" and len(all_files) < SAMPLES_PER_CLASS:
            sequences.append(augment_sequence(seq))
            labels.append(label_dict[label])

X = np.array(sequences)
y = to_categorical(np.array(labels))

print(f"\n📈 전처리 후 총 데이터 개수: {len(X)}개")
print(f"   데이터 형태: {X.shape}")
print(f"   라벨 형태: {y.shape}")

# --- 클래스 가중치 계산 및 데이터 분할 ---
y_indices = np.argmax(y, axis=1)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_indices), y=y_indices)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# --- [수정] 모델 구조 변경 (LSTM 기반) ---
# 1D CNN도 좋지만, 시계열 데이터에는 LSTM/GRU가 더 강력한 성능을 보이는 경우가 많습니다.
def create_model(learning_rate=0.001, dropout_rate=0.4):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(REQUIRED_FRAMES, EXPECTED_LEN)),
        Dropout(dropout_rate),
        LSTM(128, return_sequences=False),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --- 모델 학습 ---
# 베이지안 최적화는 시간이 매우 오래 걸리므로, 우선 검증된 값으로 학습을 시도합니다.
model = create_model()
model.summary()

history = model.fit(X_train, y_train, epochs=200, batch_size=32,
                    validation_data=(X_test, y_test),
                    class_weight=class_weight_dict,
                    callbacks=[
                        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5)
                    ])

# --- 모델 및 라벨 저장 ---
model.save(os.path.join(SAVE_DIR, "gesture_model.h5"))
label_list = [label for label, _ in sorted(label_dict.items(), key=lambda x: x[1])]
with open(os.path.join(SAVE_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_list, f, ensure_ascii=False)

# --- 성능 시각화 및 결과 확인 (기존 코드와 동일) ---
# ... (이하 시각화 및 테스트 코드) ...
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title("정확도 변화");
plt.xlabel("Epoch");
plt.ylabel("정확도");
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.title("손실 변화");
plt.xlabel("Epoch");
plt.ylabel("손실");
plt.legend()
plt.tight_layout();
plt.show()

loss, acc = model.evaluate(X_test, y_test)
print(f"\n최종 검증 정확도: {acc:.4f}")