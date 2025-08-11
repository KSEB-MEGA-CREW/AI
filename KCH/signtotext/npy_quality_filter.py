import os
import numpy as np

# 🔧 설정
TARGET_LABEL = "잘하다2"
LABEL_DIR = rf"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\output_npy\cleaned_npy\{TARGET_LABEL}"
MIN_FRAMES = 6
MAX_ZERO_RATIO = 0.5

# 🔍 분석 결과 저장 리스트
file_stats = []

print(f"🔍 분석 대상 폴더: {LABEL_DIR}")
print("--------------------------------------------------")

for fname in os.listdir(LABEL_DIR):
    if not fname.endswith(".npy"):
        continue

    fpath = os.path.join(LABEL_DIR, fname)
    try:
        data = np.load(fpath)
        frame_count = data.shape[0]
        total_elements = np.prod(data.shape)
        zero_ratio = np.count_nonzero(data == 0.0) / total_elements

        file_stats.append((fname, frame_count, zero_ratio))

    except Exception as e:
        print(f"❌ 오류 발생: {fname} ({e})")

# 🔽 프레임 수 기준 정렬
file_stats_sorted = sorted(file_stats, key=lambda x: x[1])  # frame_count 기준 오름차순

# ✅ 품질 나쁜 파일 필터링
bad_files = [(f, fc, zr) for f, fc, zr in file_stats_sorted if fc < MIN_FRAMES or zr > MAX_ZERO_RATIO]

print(f"\n📊 전체 파일 수: {len(file_stats)}")
print(f"🧹 제거 대상 파일 수: {len(bad_files)}")

# 📌 제거 대상 출력
for fname, frame_count, zero_ratio in bad_files:
    print(f"🗑 {fname} | 프레임: {frame_count} | 0비율: {zero_ratio:.2f}")

# ❓ 삭제 여부 선택
delete = input("\n❗ 위 파일들을 삭제할까요? (y/n): ").strip().lower()
if delete == 'y':
    for fname, _, _ in bad_files:
        os.remove(os.path.join(LABEL_DIR, fname))
    print(f"\n✅ 총 {len(bad_files)}개 파일 삭제 완료")
else:
    print("🚫 삭제하지 않고 종료합니다.")