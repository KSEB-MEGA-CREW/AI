import os
import numpy as np
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from bayes_opt import BayesianOptimization
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False

# 🔸 경로 설정
DATA_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\want_output_npy"
SAVE_DIR = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\models\test4"
os.makedirs(SAVE_DIR, exist_ok=True)

REQUIRED_FRAMES = 10
EXPECTED_LEN = 194
MIN_VALID_FRAMES = 6
MAX_PADDING_RATIO = 0.5

# ===== 데이터 로딩 및 사용자 입력 =====
label_files = defaultdict(list)
error_files = []

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".npy"):
            try:
                if file.startswith("none"):
                    label = "none"
                else:
                    name_split = file.split("_")
                    label = name_split[1].split(".")[0] if len(name_split) >= 2 else None
                if label:
                    label_files[label].append(os.path.join(root, file))
                else:
                    error_files.append(os.path.join(root, file))
            except Exception as e:
                print(f"❌ 파일명 파싱 오류: {file} ({e})")
                error_files.append(os.path.join(root, file))

label_count_list = sorted(label_files.items(), key=lambda x: len(x[1]), reverse=True)
print("\n📊 라벨별 npy 개수:")
for i, (label, files) in enumerate(label_count_list, 1):
    print(f"{i:3d}. {label:15s}: {len(files)}개")

file_counts = [len(files) for files in label_files.values()]
print(f"\n총 라벨 수: {len(label_files)}개, 총 npy 파일 수: {sum(file_counts)}개")
if error_files:
    print(f"\n[⚠️ 파싱 불가 파일]: {error_files}")

# ===== 사용자 입력 =====
try:
    TOP_N = int(input("\n👉 학습할 라벨 개수(예: 30): ").strip())
except Exception:
    TOP_N = 30
print(f"✅ 학습할 라벨 개수: {TOP_N}")

try:
    MIN_SAMPLES = int(input("👉 라벨별 최소 데이터 개수 이상만 포함(예: 30): ").strip())
except Exception:
    MIN_SAMPLES = 30
print(f"✅ 라벨별 최소 데이터 개수: {MIN_SAMPLES}")

eligible_labels = [label for label, files in label_files.items() if len(files) >= MIN_SAMPLES]
sorted_labels = sorted([(label, label_files[label]) for label in eligible_labels], key=lambda x: len(x[1]), reverse=True)

if len(eligible_labels) < TOP_N:
    raise ValueError(f"⚠️ MIN_SAMPLES={MIN_SAMPLES} 기준을 만족하는 라벨이 {len(eligible_labels)}개뿐입니다.")

selected_labels = [label for label, files in sorted_labels[:TOP_N]]
label_dict = {label: i for i, label in enumerate(selected_labels)}
print(f"\n✅ 최종 학습 라벨 목록 ({TOP_N}개):\n{selected_labels}")

# ===== 데이터 전처리 =====
sequences, labels = [], []
for label in selected_labels:
    files = random.sample(label_files[label], MIN_SAMPLES)
    for file in files:
        seq = np.load(file)
        if seq.shape[0] < MIN_VALID_FRAMES:
            continue
        if seq.shape[0] < REQUIRED_FRAMES:
            pad = np.zeros((REQUIRED_FRAMES - seq.shape[0], EXPECTED_LEN))
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:REQUIRED_FRAMES]
        if np.sum(seq == 0) / seq.size > MAX_PADDING_RATIO:
            continue
        max_abs = np.max(np.abs(seq))
        if max_abs > 0:
            seq = seq / max_abs
        sequences.append(seq)
        labels.append(label_dict[label])

X = np.array(sequences)
y = to_categorical(labels)

# ===== 클래스 가중치 계산 =====
y_labels = np.array(labels)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_labels), y=y_labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== 베이지안 최적화 =====
def cnn_eval(learning_rate, dropout1, dropout2):
    model = Sequential([
        Conv1D(128, 7, activation='relu', padding='same', input_shape=(REQUIRED_FRAMES, EXPECTED_LEN)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout1),
        Conv1D(256, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(dropout1),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=60, batch_size=16,
              validation_data=(X_test, y_test),
              callbacks=[
                  EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                  ReduceLROnPlateau(monitor='val_loss', patience=4)
              ],
              verbose=0)
    return model.evaluate(X_test, y_test, verbose=0)[1]

bo = BayesianOptimization(
    f=cnn_eval,
    pbounds={'learning_rate': (1e-4, 3e-3), 'dropout1': (0.1, 0.5), 'dropout2': (0.1, 0.5)},
    random_state=42
)
bo.maximize(init_points=5, n_iter=10)

# ===== 최적 파라미터로 모델 학습 =====
best = bo.max['params']
model = Sequential([
    Conv1D(128, 7, activation='relu', padding='same', input_shape=(REQUIRED_FRAMES, EXPECTED_LEN)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(best['dropout1']),
    Conv1D(256, 5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(best['dropout2']),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(best['dropout1']),
    Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer=Adam(best['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=1000, batch_size=16,
                    validation_data=(X_test, y_test),
                    class_weight=class_weight_dict,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5)
                    ])

# ===== 모델 및 라벨 저장 =====
model.save(os.path.join(SAVE_DIR, "gesture_model.h5"))
label_list = [label for label, _ in sorted(label_dict.items(), key=lambda x: x[1])]
with open(os.path.join(SAVE_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_list, f, ensure_ascii=False)

# ===== 성능 시각화 =====
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title("정확도 변화"); plt.xlabel("Epoch"); plt.ylabel("정확도"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.title("손실 변화"); plt.xlabel("Epoch"); plt.ylabel("손실"); plt.legend()
plt.tight_layout()
plt.show()

# ===== 예측 결과 확인 (최대 20개)
y_pred = model.predict(X_test)
y_pred_label = np.argmax(y_pred, axis=1)
y_true_label = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred_label == y_true_label)
print(f"\n[OFFLINE TEST] 모델 Test셋 정확도: {accuracy:.4f}")
for i in range(min(20, len(y_true_label))):
    gt = label_list[y_true_label[i]]
    pred = label_list[y_pred_label[i]]
    print(f"[{i:02d}] 실제: {gt:10s} | 예측: {pred:10s}")