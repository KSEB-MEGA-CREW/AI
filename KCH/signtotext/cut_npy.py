import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

# ====== 사용자 설정 ======
INPUT_DIR = r"C:\want_npy"  # 원본 npy 폴더
OUTPUT_DIR = r"C:\cleaned_npy"  # 정제된 결과 저장 폴더
STD_THRESHOLD = 0.005
MEAN_THRESHOLD = 0.005  # 평균 기준 (너무 낮으면 튄 것 간과함)
MIN_VALID_FRAMES = 3  # 최소 유효 프레임 수


# ====== 프레임 유효성 판단 ======
def is_frame_valid(frame):
    left = np.array(frame[:21*3])
    right = np.array(frame[21*3:21*6])
    pose = np.array(frame[21*6:])

    left_std, left_mean = np.std(left), np.mean(left)
    right_std, right_mean = np.std(right), np.mean(right)
    pose_std, pose_mean = np.std(pose), np.mean(pose)

    return (
        left_std > STD_THRESHOLD and left_mean > MEAN_THRESHOLD and
        right_std > STD_THRESHOLD and right_mean > MEAN_THRESHOLD and
        pose_std > STD_THRESHOLD and pose_mean > MEAN_THRESHOLD
    )

# ====== 정제 함수 ======
def clean_npy_file(input_path, output_path, visualize=False):
    data = np.load(input_path)
    cleaned = []
    invalid_indices = []

    for idx, frame in enumerate(data):
        if is_frame_valid(frame):
            cleaned.append(frame)
        else:
            invalid_indices.append(idx)

    if len(cleaned) < MIN_VALID_FRAMES:
        print(f"⚠️ {os.path.basename(input_path)}: 유효 프레임 부족 → 스킵됨")
        return False

    np.save(output_path, np.array(cleaned))
    print(f"✅ {os.path.basename(input_path)}: {len(data)} → {len(cleaned)} 프레임 저장됨")

    if visualize:
        plt.figure(figsize=(10, 2))
        plt.title(f"{os.path.basename(input_path)} 튄 프레임 시각화")
        plt.plot([1 if i not in invalid_indices else 0 for i in range(len(data))], marker='o')
        plt.xlabel("프레임 인덱스")
        plt.ylabel("정상(1) / 튐(0)")
        plt.grid(True)
        plt.show()

    return True


# ====== 전체 폴더 순회 및 저장 ======
for label in os.listdir(INPUT_DIR):
    label_input_path = os.path.join(INPUT_DIR, label)
    label_output_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(label_output_path, exist_ok=True)

    for fname in os.listdir(label_input_path):
        if not fname.endswith(".npy"):
            continue

        in_path = os.path.join(label_input_path, fname)
        out_path = os.path.join(label_output_path, fname)

        success = clean_npy_file(in_path, out_path, visualize=False)
        if not success:
            # 너무 튄 경우 원본 파일을 복사해도 되고, 무시해도 됨
            pass