import numpy as np
import matplotlib.pyplot as plt
import time

# 🔹 시각화할 .npy 파일 경로
npy_file = 'output_keypoints.npy'
data = np.load(npy_file)

print(f"🔍 데이터 shape: {data.shape}")  # (프레임 수, 194)

# 🔹 각 파트의 인덱스 범위
LH_IDX = slice(0, 63)      # 왼손 (21점 × 3)
RH_IDX = slice(63, 126)    # 오른손
POSE_IDX = slice(126, 194) # 포즈 (17점 × 4)

# 🔹 프레임 시각화 함수 (수치 포함)
def visualize_frame(frame_data, frame_index=0):
    keypoints = frame_data

    lh = np.array(keypoints[LH_IDX]).reshape(-1, 3)
    rh = np.array(keypoints[RH_IDX]).reshape(-1, 3)
    pose = np.array(keypoints[POSE_IDX]).reshape(-1, 4)

    print(f"\n📦 [Frame {frame_index}]")
    print(f"▶ Left Hand:\n{lh}")
    print(f"▶ Right Hand:\n{rh}")
    print(f"▶ Pose:\n{pose}")

    # 시각화 (x, y 좌표만 사용)
    plt.figure(figsize=(6, 6))
    plt.title(f"Frame {frame_index}")
    plt.xlim(0, 1)
    plt.ylim(1, 0)  # y축 반전
    plt.axis('off')

    if lh.size > 0:
        plt.scatter(lh[:, 0], lh[:, 1], c='red', label='Left Hand')
    if rh.size > 0:
        plt.scatter(rh[:, 0], rh[:, 1], c='blue', label='Right Hand')
    if pose.size > 0:
        plt.scatter(pose[:, 0], pose[:, 1], c='green', label='Pose')

    plt.legend()
    plt.show()

# 🔁 전체 프레임 순차 시각화
for idx in range(len(data)):
    visualize_frame(data[idx], frame_index=idx)
    time.sleep(0.2)  # 각 프레임 사이 딜레이 (초)