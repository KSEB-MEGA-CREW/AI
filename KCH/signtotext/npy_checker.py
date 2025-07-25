import numpy as np
import matplotlib.pyplot as plt
import time

# 1️⃣ 파일 경로 입력
npy_file = r'C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\output_npy\test\VXPAKOKS240779230_무엇1.npy'  # ← 본인 npy 경로로 수정

# 2️⃣ 데이터 로드
data = np.load(npy_file)
print(f"🔍 데이터 shape: {data.shape}")  # (프레임 수, feature 수)

# 3️⃣ 파트별 인덱스 슬라이스
LH_IDX = slice(0, 63)       # 왼손 (21점 × 3)
RH_IDX = slice(63, 126)     # 오른손 (21점 × 3)
POSE_IDX = slice(126, 194)  # 포즈 (17점 × 4)

# 4️⃣ 프레임별 값, 시각화 함수
def visualize_frame(frame_data, frame_index=0):
    lh = np.array(frame_data[LH_IDX]).reshape(-1, 3)
    rh = np.array(frame_data[RH_IDX]).reshape(-1, 3)
    pose = np.array(frame_data[POSE_IDX]).reshape(-1, 4)

    print(f"\n📦 [Frame {frame_index}]")
    print(f"▶ Left Hand shape: {lh.shape}\n{lh}")
    print(f"▶ Right Hand shape: {rh.shape}\n{rh}")
    print(f"▶ Pose shape: {pose.shape}\n{pose}")

    # x, y만 시각화
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

# 5️⃣ 전체 프레임 순차 시각화 (frame 개수 많으면 idx 범위 제한 추천)
for idx in range(min(len(data), 40)):  # 최대 30프레임만 예시
    visualize_frame(data[idx], frame_index=idx)
    time.sleep(0.2)