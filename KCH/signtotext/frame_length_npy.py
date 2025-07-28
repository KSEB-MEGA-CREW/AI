import os
import numpy as np

# 🔧 설정
target_dir = r"C:\want_output_npy"  # gloss별 폴더들이 들어있는 최상위 디렉토리

file_lengths = []

# 🔁 모든 하위 폴더 탐색
for root, _, files in os.walk(target_dir):
    for fname in files:
        if fname.endswith(".npy"):
            fpath = os.path.join(root, fname)
            try:
                arr = np.load(fpath)
                frame_count = arr.shape[0]
                file_lengths.append((fpath, frame_count))
            except Exception as e:
                print(f"❌ 오류 발생: {fpath} → {e}")

# 📊 프레임 수 기준 정렬
file_lengths.sort(key=lambda x: x[1], reverse=True)

# 📌 결과 출력
print("📋 프레임 수가 많은 순으로 정렬된 .npy 파일:")
for path, length in file_lengths:
    print(f"{length:>4}프레임 - {path}")