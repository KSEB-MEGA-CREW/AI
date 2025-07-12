import os
from collections import Counter

DATA_PATH = "output_npy"
label_counts = Counter()

for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        try:
            # 예: VXPAKOKS240779230_공부1_C.npy → "공부1"
            label = file.split("_")[1].split('.')[0]
            label_counts[label] += 1
        except Exception as e:
            print(f"파일명 파싱 오류: {file}")

# 결과 출력
for label, count in label_counts.most_common():
    print(f"{label}: {count}개")
print(f"\n총 라벨 종류: {len(label_counts)}개, 총 npy 개수: {sum(label_counts.values())}개")