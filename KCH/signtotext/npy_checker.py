import os
import numpy as np
import matplotlib.pyplot as plt
import time

# 1️⃣ 파일 경로 입력
npy_file = r'C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\output_npy\before\경제1\VXPAKOKS241130950_경제1.npy'

if not os.path.exists(npy_file):
    raise FileNotFoundError(f"❌ 파일이 존재하지 않습니다: {npy_file}")

# 2️⃣ 데이터 로드
data = np.load(npy_file)
print(f"🔍 데이터 shape: {data.shape}")  # (프레임 수, feature 수)
print(f"\n📊 총 프레임 수: {len(data)}")
zero_ratios = [np.count_nonzero(frame == 0.0) / frame.size for frame in data]
print(f"📉 프레임별 0 비율 평균: {np.mean(zero_ratios):.2f}, 최대: {np.max(zero_ratios):.2f}")

# 3️⃣ 파트별 인덱스 슬라이스
LH_IDX = slice(0, 63)
RH_IDX = slice(63, 126)
POSE_IDX = slice(126, 194)

# 4️⃣ 프레임 시각화 함수
def visualize_frame(frame_data, frame_index=0):
    lh = np.array(frame_data[LH_IDX]).reshape(-1, 3)
    rh = np.array(frame_data[RH_IDX]).reshape(-1, 3)
    pose = np.array(frame_data[POSE_IDX]).reshape(-1, 4)

    print(f"\n📦 [Frame {frame_index}]")
    print(f"▶ Left Hand shape: {lh.shape}\n{lh}")
    print(f"▶ Right Hand shape: {rh.shape}\n{rh}")
    print(f"▶ Pose shape: {pose.shape}\n{pose}")

    plt.figure(figsize=(5, 5))
    plt.title(f"Frame {frame_index}")
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.axis('off')

    if lh.size > 0:
        plt.scatter(lh[:, 0], lh[:, 1], c='red', label='Left Hand')
    if rh.size > 0:
        plt.scatter(rh[:, 0], rh[:, 1], c='blue', label='Right Hand')
    if pose.size > 0:
        plt.scatter(pose[:, 0], pose[:, 1], c='green', label='Pose')

    plt.legend()
    plt.show()

# 5️⃣ 시각화 시작
MAX_FRAMES = 30
for idx in range(min(len(data), MAX_FRAMES)):
    visualize_frame(data[idx], frame_index=idx)
    time.sleep(0.2)