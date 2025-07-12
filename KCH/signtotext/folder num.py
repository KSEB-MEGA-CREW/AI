import os
from collections import Counter

DATA_PATH = "output_npy"  # 폴더명 본인 환경에 맞게 조정

label_counts = Counter()
for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        # 예: VXPAKOKS240779230_일하다1_C.npy  → label = '일하다1'
        try:
            label = file.split("_")[1].split('.')[0]
            label_counts[label] += 1
        except Exception as e:
            print(f"❌ 파일명 파싱 오류: {file} ({e})")

# 출력
print("라벨별 npy 개수:")
for label, count in label_counts.most_common():
    print(f"{label}: {count}개")
print(f"\n총 라벨 수: {len(label_counts)}개")
print(f"총 npy 파일 수: {sum(label_counts.values())}개")