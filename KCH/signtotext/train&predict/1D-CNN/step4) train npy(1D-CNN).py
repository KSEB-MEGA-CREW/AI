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

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\output_npy\test"
REQUIRED_FRAMES = 10
EXPECTED_LEN = 194
MIN_VALID_FRAMES = 7
MAX_PADDING_RATIO = 0.4

SAVE_DIR = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\train&predict\models\test_model"
os.makedirs(SAVE_DIR, exist_ok=True)

label_files = defaultdict(list)
for root, dirs, files in os.walk(DATA_PATH):
    for fname in files:
        if fname.endswith(".npy"):
            try:
                # ① none은 무조건 'none' 라벨, 그 외는 파일명 첫 부분
                if fname.startswith("none"):
                    label = "none"
                else:
                    label = fname.split("_")[0]
                label_files[label].append(os.path.join(root, fname))
            except Exception as e:
                print(f"❌ 파일명 파싱 오류: {fname} ({e})")

# (1) 내림차순 정렬 + 번호 출력
label_count_list = sorted(label_files.items(), key=lambda x: len(x[1]), reverse=True)
print("\n라벨별 npy 개수 (개수 많은 순, 번호순):")
for i, (label, files) in enumerate(label_count_list, 1):
    print(f"{i:3d}. {label:15s}: {len(files)}개")
print(f"\n총 라벨 수: {len(label_files)}개")
print(f"총 npy 파일 수: {sum(len(files) for _, files in label_count_list)}개")

file_counts = [len(files) for files in label_files.values()]
if file_counts:
    print(f"라벨별 최소 개수: {min(file_counts)}, 최대 개수: {max(file_counts)}, 평균: {np.mean(file_counts):.1f}, 중앙값: {np.median(file_counts):.1f}")

# [3] 사용자 입력
try:
    TOP_N = int(input("\n학습할 라벨 개수(예: 30): ").strip())
except Exception:
    TOP_N = 30
print(f"학습할 라벨 개수: {TOP_N}")

try:
    MIN_SAMPLES = int(input("라벨별 최소 데이터 개수 이상만 포함(예: 30): ").strip())
except Exception:
    MIN_SAMPLES = 30
print(f"라벨별 최소 데이터 개수: {MIN_SAMPLES}")

random.seed(42)

# [4] 라벨 필터링 (MIN_SAMPLES 이상)
eligible_labels = [label for label, files in label_files.items() if len(files) >= MIN_SAMPLES]
print(f"\n[{MIN_SAMPLES}개 이상 npy 가진 라벨 수]: {len(eligible_labels)}개")

# [5] 개수 많은 순으로 TOP-N 라벨 선정
sorted_labels = sorted([(label, label_files[label]) for label in eligible_labels], key=lambda x: len(x[1]), reverse=True)
print(f"\n[조건 만족 라벨 TOP-{TOP_N} 미리보기]")
for i, (label, files) in enumerate(sorted_labels[:TOP_N], 1):
    print(f"{i:3d}. {label:15s}: {len(files)}개")

if len(eligible_labels) < TOP_N:
    raise ValueError(f"\nMIN_SAMPLES={MIN_SAMPLES} 기준을 만족하는 라벨이 {len(eligible_labels)}개뿐입니다. 값을 조정하세요.")

selected_labels = [label for label, files in sorted_labels[:TOP_N]]
print(f"\n[최종 학습 라벨 목록 ({TOP_N}개)]")
print(selected_labels)

# [6] 데이터셋 만들기 (각 라벨별 MIN_SAMPLES 랜덤 추출)
sequences, labels = [], []
label_dict = {label: i for i, label in enumerate(selected_labels)}

for label in selected_labels:
    files = label_files[label]
    chosen_files = random.sample(files, MIN_SAMPLES)
    label_num = label_dict[label]
    for file in chosen_files:
        sequence = np.load(file)  # 이미 풀경로임!
        if sequence.shape[0] < MIN_VALID_FRAMES:
            continue

        # 프레임 수 맞춤 (패딩/자르기)
        if sequence.shape[0] < REQUIRED_FRAMES:
            pad = np.zeros((REQUIRED_FRAMES - sequence.shape[0], EXPECTED_LEN))
            sequence_fixed = np.vstack([sequence, pad])
        else:
            sequence_fixed = sequence[:REQUIRED_FRAMES]

        # 패딩 비율 계산 (0이 전체의 몇 %인지)
        zero_ratio = np.sum(sequence_fixed == 0) / sequence_fixed.size
        if zero_ratio > MAX_PADDING_RATIO:
            continue  # 패딩이 너무 많으면 학습 제외

        # 정규화 (최대 절대값 기준)
        max_abs = np.max(np.abs(sequence_fixed))
        if max_abs > 0:
            sequence_fixed = sequence_fixed / max_abs
        sequences.append(sequence_fixed)
        labels.append(label_num)

X = np.array(sequences)
y = to_categorical(labels)
print(f"\n[최종 데이터 shape] X={X.shape}, y={y.shape}, 라벨={len(label_dict)}개")

# [7] 학습/검증 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# [8] Bayesian Optimization 대상 함수 정의
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
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=60, # 속도 위해 epoch 제한
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    score = model.evaluate(X_test, y_test, verbose=0)
    val_acc = score[1]
    return val_acc

# [9] 베이지안 최적화 범위 지정 및 실행
pbounds = {
    'learning_rate': (1e-4, 3e-3),
    'dropout1': (0.1, 0.5),
    'dropout2': (0.1, 0.5)
}

bo = BayesianOptimization(
    f=cnn_train_eval,
    pbounds=pbounds,
    random_state=42
)

print("\n[Bayesian Optimization] CNN 하이퍼파라미터 탐색 시작")
bo.maximize(init_points=5, n_iter=12)

# [10] 최적 파라미터로 최종 모델 학습
best_params = bo.max['params']
print(f"\n[최적 파라미터] {best_params}")

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

from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=1000,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# [11] 모델 및 라벨맵 저장 (⭐️ 원하는 폴더에 저장)
model.save(os.path.join(SAVE_DIR, "gesture_model.h5"))
label_list = [label for label, idx in sorted(label_dict.items(), key=lambda x: x[1])]
with open(os.path.join(SAVE_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_list, f, ensure_ascii=False)

# [12] 학습 곡선 시각화
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

# [13] Test셋 오프라인 예측 및 정확도 체크
print("\n[OFFLINE TEST] 모델 Test셋 예측 결과:")
y_pred = model.predict(X_test)
y_pred_label = np.argmax(y_pred, axis=1)
y_true_label = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred_label == y_true_label)
print(f"\n[OFFLINE TEST] 모델 Test셋 정확도: {accuracy:.4f}")

for i in range(min(20, len(y_true_label))):
    gt = label_list[y_true_label[i]]
    pred = label_list[y_pred_label[i]]
    print(f"[{i:02d}] 실제: {gt:10s} | 예측: {pred:10s}")