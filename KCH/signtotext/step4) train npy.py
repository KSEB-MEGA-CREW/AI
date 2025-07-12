import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout, Add, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib
import json

# 🔹 한글 폰트 설정 (그래프용)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

MIN_COUNT = 1
DATA_PATH = "output_npy"
label_counts = Counter()
for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        try:
            label = file.split("_")[1].split('.')[0]
            label_counts[label] += 1
        except Exception as e:
            print(f"❌ 파일명 파싱 오류: {file}")

selected_labels = {label for label, count in label_counts.items() if count >= MIN_COUNT}
print(f"✅ {MIN_COUNT}개 이상 라벨만 학습에 사용: {selected_labels}")

sequences = []
labels = []
label_dict = {}
label_idx = 0

expected_len = 194
required_frames = 24

def jitter(seq, sigma=0.01):
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise

def window_sequences(seq, win_size=24, stride=4):
    seqs = []
    for start in range(0, len(seq) - win_size + 1, stride):
        seqs.append(seq[start:start+win_size])
    # 만약 길이가 충분하지 않으면 마지막 window 한 번 추가 (padding)
    if len(seq) < win_size:
        pad = np.zeros((win_size - len(seq), seq.shape[1]))
        seqs.append(np.vstack([seq, pad]))
    return seqs

for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        try:
            label = file.split("_")[1].split('.')[0]
            if label not in selected_labels:
                continue
            path = os.path.join(DATA_PATH, file)
            sequence = np.load(path)
            windows = window_sequences(sequence, win_size=required_frames, stride=4)
            if label not in label_dict:
                label_dict[label] = label_idx
                label_idx += 1
            label_num = label_dict[label]
            for seq_win in windows:
                sequences.append(seq_win)
                labels.append(label_num)
                # 증강 데이터도 추가 (jitter)
                sequences.append(jitter(seq_win))
                labels.append(label_num)
        except Exception as e:
            print(f"❌ 오류: {file} → {e}")

X = np.array(sequences)
y = to_categorical(labels)
print(f"✅ 데이터 로딩 완료! X shape: {X.shape}, y shape: {y.shape}")
print(f"📄 라벨 매핑: {label_dict}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 🔹 class_weight 계산 (불균형 보정)
label_integers = np.argmax(y_train, axis=1)
cw = compute_class_weight('balanced', classes=np.unique(label_integers), y=label_integers)
class_weight = {i: cw[i] for i in range(len(cw))}
print(f"Class weight: {class_weight}")

# ------------- Residual Block 구현 함수 -------------
def residual_block(x, filters, kernel_size, dilation, dropout_rate):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation)(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    return x

# ------------- 모델 구성 -------------
input_layer = Input(shape=(X.shape[1], expected_len))
x = Conv1D(128, 7, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.18)(x)

x = residual_block(x, 128, 3, dilation=2, dropout_rate=0.18)
x = MaxPooling1D(2)(x)
x = residual_block(x, 128, 3, dilation=4, dropout_rate=0.22)
x = MaxPooling1D(2)(x)

x = Conv1D(256, 3, activation='relu', padding='same', dilation_rate=2)(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.22)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.18)(x)
output = Dense(y.shape[1], activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=18, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=90,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
    class_weight=class_weight
)

os.makedirs("models", exist_ok=True)
model.save("models/gesture_model.h5")
print("✅ 모델 저장 완료: models/gesture_model.h5")

sorted_labels = [label for label, idx in sorted(label_dict.items(), key=lambda x: x[1])]
with open("models/label_map.json", "w", encoding="utf-8") as f:
    json.dump(sorted_labels, f, ensure_ascii=False)
print("✅ label_map.json 저장 완료!")

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