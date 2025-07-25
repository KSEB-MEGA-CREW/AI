import os
from collections import Counter

DATA_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\output_npy\일상생활_학교"

label_counts = Counter()
error_files = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".npy"):
            try:
                name_split = file.split("_")
                if len(name_split) >= 2:
                    label_part = name_split[1]
                    label = label_part.split(".")[0]
                    label_counts[label] += 1
                else:
                    error_files.append(os.path.join(root, file))
            except Exception as e:
                print(f"❌ 파일명 파싱 오류: {file} ({e})")
                error_files.append(os.path.join(root, file))

# [★] 라벨별 개수 내림차순 정렬
sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

print("라벨별 npy 개수 (개수 많은 순, 번호순):")
for i, (label, count) in enumerate(sorted_labels, 1):
    print(f"{i:3d}. {label:15s}: {count}개")

print(f"\n총 라벨 수: {len(label_counts)}개")
print(f"총 npy 파일 수: {sum(label_counts.values())}개")
if error_files:
    print(f"\n[⚠️ 파싱 불가 파일]: {error_files}")