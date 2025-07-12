import os
import json
from collections import Counter

# ① 폴더 경로 지정 (절대경로 사용!)
folder = r"C:\KEB_bootcamp\project\AI/KCH/signtotext/NIKL_Sign Language Parallel Corpus_2024_LI_CO"
gloss_counter = Counter()

# ② 모든 하위폴더 포함 json파일 탐색
for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".json"):
            json_path = os.path.join(root, file)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])
            for gloss in gestures:
                gloss_id = str(gloss.get("gloss_id", "")).replace('.npy', '').replace('.NPY', '')
                gloss_counter[gloss_id] += 1

# ③ 결과 출력
print("라벨별 gloss 데이터 개수 (등장 횟수 기준):")
for gloss_id, count in gloss_counter.most_common():
    print(f"{gloss_id}: {count}개")
print(f"\n총 라벨 종류: {len(gloss_counter)}개, 총 gloss 개수: {sum(gloss_counter.values())}개")
