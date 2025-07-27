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
from bayes_opt import BayesianOptimization
from scipy.interpolate import interp1d

# ===== [시각화 설정] =====
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== [기본 설정] =====
DATA_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\output_npy\before"
REQUIRED_FRAMES = 12
EXPECTED_LEN = 194
MIN_VALID_FRAMES = 7
MAX_PADDING_RATIO = 0.4
SAVE_DIR = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\train&predict\1D-CNN\models\보간\테스트"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== [보간 함수] =====
def interpolate_sequence(sequence, target_len=REQUIRED_FRAMES):
    current_len = sequence.shape[0]
    if current_len == target_len:
        return sequence
    x_old = np.linspace(0, 1, num=current_len)
    x_new = np.linspace(0, 1, num=target_len)
    interpolated = interp1d(x_old, sequence, axis=0, kind='linear', fill_value="extrapolate")(x_new)
    return interpolated

# ===== [보간 디버깅 함수] =====
def debug_interpolation(sequence, target_len=REQUIRED_FRAMES, title=None):
    interpolated = interpolate_sequence(sequence, target_len)
    original_x = sequence[:, 0]  # 예: 0번 keypoint의 x좌표
    interpolated_x = interpolated[:, 0]

    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, 1, len(original_x)), original_x, marker='o', label=f"원본 ({len(original_x)}프레임)")
    plt.plot(np.linspace(0, 1, len(interpolated_x)), interpolated_x, marker='x', linestyle='--', label=f"보간 ({target_len}프레임)")
    plt.title(title or "보간 시각화 (0번 keypoint x좌표 기준)")
    plt.xlabel("정규화된 시간")
    plt.ylabel("값")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ===== [데이터 로딩 및 라벨 처리] =====
label_files = defaultdict(list)
error_files = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".npy"):
            try:
                if file.startswith("none"):
                    label = "none"
                else:
                    name_split = file.split("_")
                    if len(name_split) >= 2:
                        label_part = name_split[1]
                        label = label_part.split(".")[0]
                    else:
                        error_files.append(os.path.join(root, file))
                        continue
                label_files[label].append(os.path.join(root, file))
            except Exception as e:
                print(f"❌ 파일명 파싱 오류: {file} ({e})")
                error_files.append(os.path.join(root, file))

label_count_list = sorted(label_files.items(), key=lambda x: len(x[1]), reverse=True)
print("\n라벨별 npy 개수:")
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
    raise ValueError(f"MIN_SAMPLES={MIN_SAMPLES} 기준을 만족하는 라벨이 {len(eligible_labels)}개뿐입니다.")
selected_labels = [label for label, files in sorted_labels[:TOP_N]]
print(f"\n[최종 학습 라벨 목록 ({TOP_N}개)]\n{selected_labels}")

# ===== [데이터 전처리 및 디버깅 시각화 일부 포함] =====
sequences, labels = [], []
label_dict = {label: i for i, label in enumerate(selected_labels)}

for label in selected_labels:
    files = label_files[label]
    chosen_files = random.sample(files, MIN_SAMPLES)
    label_num = label_dict[label]
    for idx, file in enumerate(chosen_files):
        sequence = np.load(file)
        if sequence.shape[0] < MIN_VALID_FRAMES:
            continue
        if idx == 0:  # 각 라벨당 첫 번째 샘플 시각화
            print(f"\n📊 [디버깅 시각화] 라벨: {label} ({file})")
            debug_interpolation(sequence, target_len=REQUIRED_FRAMES, title=f"{label} 보간 디버깅")

        sequence_fixed = interpolate_sequence(sequence, REQUIRED_FRAMES)
        zero_ratio = np.sum(sequence_fixed == 0) / sequence_fixed.size
        if zero_ratio > MAX_PADDING_RATIO:
            continue
        max_abs = np.max(np.abs(sequence_fixed))
        if max_abs > 0:
            sequence_fixed = sequence_fixed / max_abs
        sequences.append(sequence_fixed)
        labels.append(label_num)

X = np.array(sequences)
y = to_categorical(labels)
print(f"\n[최종 데이터 shape] X={X.shape}, y={y.shape}, 라벨 수={len(label_dict)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== [모델 최적화 및 학습] =====
def cnn_train_eval(learning_rate, dropout1, dropout2):
    from tensorflow.keras.optimizers import Adam
    model = Sequential([
        Conv1D(128, 7, activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2])),
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
        Dense(256, activation='relu'),
        Dropout(dropout2),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=0)
    score = model.evaluate(X_test, y_test, verbose=0)
    return score[1]

pbounds = {
    'learning_rate': (1e-4, 3e-3),
    'dropout1': (0.1, 0.5),
    'dropout2': (0.1, 0.5)
}

print("\n[Bayesian Optimization] 하이퍼파라미터 탐색 시작")
bo = BayesianOptimization(f=cnn_train_eval, pbounds=pbounds, random_state=42)
bo.maximize(init_points=5, n_iter=12)
best_params = bo.max['params']
print(f"\n[최적 파라미터] {best_params}")

# ===== [최종 모델 학습 및 저장] =====
from tensorflow.keras.optimizers import Adam
model = Sequential([
    Conv1D(128, 7, activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(best_params['dropout1']),
    Conv1D(256, 5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(best_params['dropout2']),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(best_params['dropout1']),
    Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=1)

model.save(os.path.join(SAVE_DIR, "gesture_model.h5"))
label_list = [label for label, idx in sorted(label_dict.items(), key=lambda x: x[1])]
with open(os.path.join(SAVE_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_list, f, ensure_ascii=False)

# ===== [성능 시각화 및 평가] =====
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

y_pred = model.predict(X_test)
y_pred_label = np.argmax(y_pred, axis=1)
y_true_label = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred_label == y_true_label)
print(f"\n[OFFLINE TEST] 모델 Test셋 정확도: {accuracy:.4f}")
for i in range(min(20, len(y_true_label))):
    gt = label_list[y_true_label[i]]
    pred = label_list[y_pred_label[i]]
    print(f"[{i:02d}] 실제: {gt:10s} | 예측: {pred:10s}")