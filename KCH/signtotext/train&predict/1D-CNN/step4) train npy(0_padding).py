import os
import json
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# 선택: bayes_opt 미설치 환경 대비
try:
    from bayes_opt import BayesianOptimization
    HAS_BO = True
except Exception:
    HAS_BO = False

# =============================
# 기본 설정
# =============================
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
np.set_printoptions(suppress=True)

# 🔸 경로 설정 (필요 시 변경)
DATA_PATH = r"C:\\want_npy_v2"                     # 라벨/시나리오별 폴더 구조 하의 .npy
SAVE_DIR  = r"C:\\models\\v5_cnn_bo_reports[14]"   # 모델/리포트 산출물 저장 폴더
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔸 전처리/모델 기본 설정
REQUIRED_FRAMES = 10      # 시퀀스 길이(프레임 수)
EXPECTED_LEN    = 194     # 프레임당 feature 길이 (채널 수)
TOP_N           = 17      # (none 제외) 상위 라벨 개수
SAMPLES_PER_CLASS = 293   # 각 라벨 최대 사용 샘플 수
INCLUDE_NONE    = True    # 'none' 라벨 자동 포함
# (선택) none만 별도 상한을 두고 싶으면 아래 값을 바꾸세요. None이면 SAMPLES_PER_CLASS와 동일하게 처리.
SAMPLES_PER_CLASS_NONE = None

# 🔸 Bayesian Optimization 설정
BO_ENABLE    = True
BO_INIT      = 8
BO_ITER      = 20
BO_RANDOM_SEED = 42

# 🔸 재현성 고정
GLOBAL_SEED = 42
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

# =============================
# 유틸: 안전한 로더
# =============================
def safe_load_sequence(file_path, expected_len=EXPECTED_LEN):
    try:
        arr = np.load(file_path, allow_pickle=True)
        arr = np.asarray(arr)
        if arr.dtype == object:
            try:
                arr = np.vstack(arr)
            except Exception:
                return None, "ragged_object_array"
        if arr.ndim == 1:
            if arr.size % expected_len == 0:
                arr = arr.reshape(-1, expected_len)
            else:
                return None, f"flat_length_{arr.size}_not_multiple_of_{expected_len}"
        if arr.ndim != 2 or arr.shape[1] == 0:
            return None, f"bad_shape_{arr.shape}"
        if arr.shape[1] != expected_len:
            if arr.shape[1] > expected_len:
                arr = arr[:, :expected_len]
            else:
                pad_cols = expected_len - arr.shape[1]
                arr = np.hstack([arr, np.zeros((arr.shape[0], pad_cols), dtype=arr.dtype)])
        return arr.astype(np.float32), None
    except Exception as e:
        return None, f"load_error_{type(e).__name__}"

# =============================
# (추가) 시간 리샘플: 정확히 T프레임으로 보간
# =============================
def temporal_resample(seq, T=REQUIRED_FRAMES, D=EXPECTED_LEN):
    """seq:(L,D) → (T,D) 선형보간. L=0,1 예외 처리 포함."""
    if len(seq) == 0:
        return np.zeros((T, D), dtype=np.float32)
    if len(seq) == 1:
        return np.repeat(seq.astype(np.float32), T, axis=0)

    idx = np.linspace(0, len(seq) - 1, num=T)
    i0 = np.floor(idx).astype(int)
    i1 = np.ceil(idx).astype(int)
    w = (idx - i0)[:, None].astype(np.float32)

    s0 = seq[i0].astype(np.float32)
    s1 = seq[i1].astype(np.float32)
    return (1 - w) * s0 + w * s1

# =============================
# 데이터 로딩 (시나리오→'none' 매핑)
# =============================
# 여기에 none으로 취급할 폴더 이름들을 등록하세요.
NONE_ALIASES = {
    'none',
    'empty_frame', 'hands_down', 'typing_mouse',
    'phone_usage', 'head_touch_glasses', 'look_around'
}

label_files = defaultdict(list)
for root, _, files in os.walk(DATA_PATH):
    label_raw = os.path.basename(root)
    label = 'none' if label_raw in NONE_ALIASES else label_raw  # 🔸 핵심: 시나리오들을 'none'으로 통합
    for file in files:
        if file.endswith('.npy'):
            label_files[label].append(os.path.join(root, file))

# 라벨별 개수 확인
label_count_list = sorted(label_files.items(), key=lambda x: len(x[1]), reverse=True)
print("\n📊 라벨별 npy 개수(시나리오 통합 반영):")
for i, (label, files) in enumerate(label_count_list, 1):
    print(f"{i:3d}. {label:15s}: {len(files)}개")

# 학습 라벨 목록 구성
selected_labels = [label for label, files in label_count_list if label != 'none'][:TOP_N]
if INCLUDE_NONE and 'none' in label_files:
    selected_labels.append('none')
label_dict = {label: i for i, label in enumerate(selected_labels)}
print(f"\n✅ 최종 학습 라벨 목록 ({len(selected_labels)}개):\n{selected_labels}")

# (미리 생성) 인덱스→라벨 리스트
label_list = [None]*len(selected_labels)
for name, idx in label_dict.items():
    label_list[idx] = name

# =============================
# 증강 (안전한 라벨 보존형만)
# =============================
def augment_sequence(sequence,
                     noise_level=0.003,
                     p_time_jitter=0.5,
                     p_frame_drop=0.3):
    """
    얌전한 증강:
      1) ±10% 시간 왜곡 후 원 길이 복원
      2) 무작위 1프레임 드롭 후 원 길이 복원
      3) 미세 가우시안 노이즈
    """
    seq = sequence.astype(np.float32)

    # 1) 시간 왜곡
    if np.random.rand() < p_time_jitter:
        alpha = 1.0 + np.random.uniform(-0.1, 0.1)  # ±10%
        mid_T = max(2, int(round(len(seq) * alpha)))
        mid = temporal_resample(seq, mid_T, seq.shape[1])
        seq = temporal_resample(mid, sequence.shape[0], sequence.shape[1])

    # 2) 프레임 드롭
    if len(seq) > 2 and np.random.rand() < p_frame_drop:
        drop_idx = np.random.randint(0, len(seq))
        seq2 = np.delete(seq, drop_idx, axis=0)
        seq = temporal_resample(seq2, sequence.shape[0], sequence.shape[1])

    # 3) 미세 노이즈
    seq = seq + np.random.normal(0, noise_level, seq.shape).astype(np.float32)
    return seq

# =============================
# 전처리 (리샘플 → 10프레임 고정)
# =============================
sequences, labels = [], []
skipped = []

# none의 상한 결정
none_cap = SAMPLES_PER_CLASS if SAMPLES_PER_CLASS_NONE is None else SAMPLES_PER_CLASS_NONE

for label in selected_labels:
    all_files = label_files[label]
    cap_for_label = none_cap if label == 'none' else SAMPLES_PER_CLASS
    files_to_use = random.sample(all_files, min(len(all_files), cap_for_label))

    for file_path in files_to_use:
        seq, err = safe_load_sequence(file_path, expected_len=EXPECTED_LEN)
        if err is not None:
            skipped.append((file_path, err))
            continue

        # ✅ 패딩/자르기 대신 시간 리샘플로 정확히 10프레임
        seq = temporal_resample(seq, REQUIRED_FRAMES, EXPECTED_LEN)

        sequences.append(seq)
        labels.append(label_dict[label])

        # 데이터가 적은 클래스에만 약한 증강 1개 추가 (none 제외)
        if label != 'none' and len(all_files) < SAMPLES_PER_CLASS:
            sequences.append(augment_sequence(seq))
            labels.append(label_dict[label])

X = np.array(sequences, dtype=np.float32)
y_indices = np.array(labels, dtype=np.int64)
y = to_categorical(y_indices, num_classes=len(selected_labels))

print(f"\n📈 전처리 후 총 데이터 개수: {len(X)}개")
print(f"   데이터 형태: {X.shape}  (frames={REQUIRED_FRAMES}, features={EXPECTED_LEN})")
print(f"   라벨 형태: {y.shape}")

if skipped:
    print(f"\n⚠️ 스킵한 파일: {len(skipped)}개 (아래 10개만 표시)")
    for fp, why in skipped[:10]:
        print(f"- {why}: {fp}")

# =============================
# 라벨 분포 저장(보고서용) - CSV/PNG
# =============================
unique, counts = np.unique(y_indices, return_counts=True)
dist_df = pd.DataFrame({
    'label_index': unique,
    'label_name': [label_list[i] for i in unique],
    'count': counts
}).sort_values('count', ascending=False)
dist_df.to_csv(os.path.join(SAVE_DIR, 'label_distribution.csv'), index=False, encoding='utf-8-sig')

plt.figure(figsize=(10, 6))
plt.bar(dist_df['label_name'], dist_df['count'])
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'label_distribution.png'), dpi=150)
plt.close()

# =============================
# 데이터 분할 및 클래스 가중치
# =============================
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_indices), y=y_indices)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

X_tr_all, X_te, y_tr_all, y_te, yi_tr_all, yi_te = train_test_split(
    X, y, y_indices, test_size=0.25, random_state=GLOBAL_SEED, stratify=y_indices
)
X_tr, X_va, y_tr, y_va, yi_tr, yi_va = train_test_split(
    X_tr_all, y_tr_all, yi_tr_all, test_size=0.15, random_state=GLOBAL_SEED, stratify=yi_tr_all
)

# =============================
# 모델: 1D-CNN
# =============================
def build_model(filters1, filters2, k1, k2, dropout, lr, l2):
    f1 = int(round(filters1))
    f2 = int(round(filters2))
    k1 = int(round(k1))
    k2 = int(round(k2))
    k1 = max(1, min(k1, REQUIRED_FRAMES))
    k2 = max(1, min(k2, REQUIRED_FRAMES))

    model = Sequential([
        Conv1D(f1, kernel_size=k1, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(l2), input_shape=(REQUIRED_FRAMES, EXPECTED_LEN)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(dropout),

        Conv1D(f2, kernel_size=k2, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(l2)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(dropout),

        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =============================
# Bayesian Optimization (옵션)
# =============================
best_params = {
    'filters1': 96,
    'filters2': 160,
    'k1': 3,
    'k2': 3,
    'dropout': 0.35,
    'lr': 1e-3,
    'l2': 1e-5,
    'batch_size': 32,
    'val_accuracy': None
}

if BO_ENABLE and HAS_BO:
    random.seed(BO_RANDOM_SEED); np.random.seed(BO_RANDOM_SEED); tf.random.set_seed(GLOBAL_SEED)

    def objective(filters1, filters2, k1, k2, dropout, log_lr, l2, batch_q):
        f1 = int(round(filters1)); f2 = int(round(filters2))
        kk1 = int(round(k1)); kk2 = int(round(k2))
        lr = 10 ** log_lr
        batch_choices = [16, 32, 48, 64]
        _idx = int(np.clip(np.round(batch_q), 0, len(batch_choices) - 1))
        bs = batch_choices[_idx]

        tf.random.set_seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); random.seed(GLOBAL_SEED)

        model = build_model(f1, f2, kk1, kk2, float(dropout), float(lr), float(l2))
        cb = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.5, verbose=0)
        ]
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=100, batch_size=bs, verbose=0,
            class_weight=class_weight_dict,
            callbacks=cb
        )
        return float(np.max(hist.history['val_accuracy']))

    pbounds = {
        'filters1': (48, 192),
        'filters2': (64, 256),
        'k1': (2, 5),
        'k2': (2, 5),
        'dropout': (0.2, 0.6),
        'log_lr': (-4.0, -2.3),   # 1e-4 ~ ~5e-3
        'l2': (1e-6, 5e-4),
        'batch_q': (0, 3.99)
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=BO_RANDOM_SEED,
        allow_duplicate_points=True
    )
    optimizer.maximize(init_points=BO_INIT, n_iter=BO_ITER)

    bp = optimizer.max['params']
    best_params['filters1'] = int(round(bp['filters1']))
    best_params['filters2'] = int(round(bp['filters2']))
    best_params['k1'] = int(round(bp['k1']))
    best_params['k2'] = int(round(bp['k2']))
    best_params['dropout'] = float(bp['dropout'])
    best_params['lr'] = float(10 ** bp['log_lr'])
    best_params['l2'] = float(bp['l2'])
    _batch_choices = [16, 32, 48, 64]
    _idx = int(np.clip(np.round(bp['batch_q']), 0, len(_batch_choices) - 1))
    best_params['batch_size'] = _batch_choices[_idx]
    best_params['val_accuracy'] = float(optimizer.max['target'])
else:
    print("\n[BO] 비활성화 또는 bayes_opt 미설치 — 기본 하이퍼파라미터 사용")

with open(os.path.join(SAVE_DIR, 'best_params.json'), 'w', encoding='utf-8') as f:
    json.dump(best_params, f, ensure_ascii=False, indent=2)
print("BEST PARAMS:", best_params)

# =============================
# 최종 학습
# =============================
final_model = build_model(best_params['filters1'], best_params['filters2'], best_params['k1'], best_params['k2'],
                          best_params['dropout'], best_params['lr'], best_params['l2'])
final_cb = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', patience=6, factor=0.5, verbose=1),
]

history = final_model.fit(
    X_tr_all, y_tr_all,
    validation_data=(X_te, y_te),
    epochs=200, batch_size=best_params['batch_size'], verbose=1,
    class_weight=class_weight_dict,
    callbacks=final_cb
)

# =============================
# 저장 및 보고서 산출물
# =============================
model_path = os.path.join(SAVE_DIR, 'gesture_model.h5')
final_model.save(model_path)

with open(os.path.join(SAVE_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
    json.dump(label_list, f, ensure_ascii=False, indent=2)

# 학습 이력 CSV/그래프
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(os.path.join(SAVE_DIR, 'history.csv'), index=False, encoding='utf-8-sig')

plt.figure()
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title('정확도 변화 (CNN)')
plt.xlabel('Epoch')
plt.ylabel('정확도')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'accuracy_curve.png'), dpi=150)
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.title('손실 변화 (CNN)')
plt.xlabel('Epoch')
plt.ylabel('손실')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'loss_curve.png'), dpi=150)
plt.close()

# 평가: 혼동행렬/리포트/예측 확률 저장 (보고서용)
loss, acc = final_model.evaluate(X_te, y_te, verbose=0)
print(f"\n[FINAL CNN] 테스트 정확도: {acc:.4f}")

probs = final_model.predict(X_te, verbose=0)
y_true_idx = np.argmax(y_te, axis=1)
y_pred_idx = np.argmax(probs, axis=1)

cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(label_list))))
cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
cm_df.to_csv(os.path.join(SAVE_DIR, 'confusion_matrix.csv'), encoding='utf-8-sig')

plt.figure(figsize=(8, 7))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix (CNN)')
plt.colorbar()
plt.xticks(ticks=np.arange(len(label_list)), labels=label_list, rotation=90)
plt.yticks(ticks=np.arange(len(label_list)), labels=label_list)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=160)
plt.close()

report_dict = classification_report(y_true_idx, y_pred_idx, target_names=label_list, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(SAVE_DIR, 'classification_report.csv'), encoding='utf-8-sig')

result_rows = []
for i in range(len(y_te)):
    t = int(y_true_idx[i])
    p = int(y_pred_idx[i])
    conf = float(probs[i, p])
    result_rows.append({
        'true_index': t,
        'true_label': label_list[t],
        'pred_index': p,
        'pred_label': label_list[p],
        'confidence': conf
    })
results_df = pd.DataFrame(result_rows)
results_df.to_csv(os.path.join(SAVE_DIR, 'test_predictions.csv'), index=False, encoding='utf-8-sig')

summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_path': DATA_PATH,
    'save_dir': SAVE_DIR,
    'required_frames': REQUIRED_FRAMES,
    'expected_len': EXPECTED_LEN,
    'num_classes': len(label_list),
    'classes': label_list,
    'train_samples': int(len(X_tr_all)),
    'test_samples': int(len(X_te)),
    'val_split_within_train': 0.15,
    'class_weights': class_weight_dict,
    'bayesian_optimization': {
        'enabled': bool(BO_ENABLE and HAS_BO),
        'init_points': BO_INIT,
        'n_iter': BO_ITER,
        'best_params': best_params
    },
    'final_test_accuracy': float(acc)
}
with open(os.path.join(SAVE_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n✅ 산출물 저장 목록")
for fn in [
    'gesture_model.h5',
    'label_map.json',
    'history.csv',
    'accuracy_curve.png',
    'loss_curve.png',
    'label_distribution.csv',
    'label_distribution.png',
    'confusion_matrix.csv',
    'confusion_matrix.png',
    'classification_report.csv',
    'test_predictions.csv',
    'best_params.json',
    'summary.json'
]:
    print(' -', os.path.join(SAVE_DIR, fn))

print("\n🎯 완료: (CNN) 보고서용 그래프와 CSV가 SAVE_DIR에 저장되었습니다.")