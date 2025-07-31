import cv2
import mediapipe as mp
import numpy as np
import json
import os
import re
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# 📂 경로 설정
root_folder = r"D:"   # 원본 데이터 루트 (json/mp4)
output_dir = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\want_output_npy"
os.makedirs(output_dir, exist_ok=True)

POSE_INDEXES = list(range(17))
expected_len = 21*3 + 21*3 + 17*4
BUFFER_FRAMES = 5  # MediaPipe 안정화 프레임

def sanitize_filename(name):
    return re.sub(r'[:\/\\?*<>|"]', '_', name)

def extract_landmarks(landmarks, dims, idxs=None):
    result = []
    if landmarks:
        iter_landmarks = (
            enumerate(landmarks.landmark) if idxs is None
            else ((i, landmarks.landmark[i]) for i in idxs)
        )
        for _, lm in iter_landmarks:
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, 'visibility', 0.0))
            result.extend(coords)
    return result

# 🔍 전체 폴더 내 json/mp4 탐색 (하위 폴더 포함)
all_json_paths = []
for dirpath, _, filenames in os.walk(root_folder):
    for fname in filenames:
        if fname.endswith(".json"):
            base = os.path.splitext(fname)[0]
            json_path = os.path.join(dirpath, f"{base}.json")
            video_path = os.path.join(dirpath, f"{base}.mp4")
            if os.path.exists(video_path):
                all_json_paths.append((base, json_path, video_path))

print(f"🔍 총 {len(all_json_paths)}개 JSON/MP4 세트 발견됨")

# [1] 데이터 내 전체 라벨 현황 집계 (병렬 집계)
def count_gloss_ids_worker(json_path):
    counter = Counter()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for gloss in data.get("sign_script", {}).get("sign_gestures_strong", []):
            gloss_id = str(gloss['gloss_id']).strip()
            counter[gloss_id] += 1
    return counter

label_counter = Counter()
with ThreadPoolExecutor(max_workers=12) as executor:
    for c in executor.map(lambda x: count_gloss_ids_worker(x[1]), all_json_paths):
        label_counter.update(c)

label_count_list = sorted(label_counter.items(), key=lambda x: x[1], reverse=True)
print("\n📊 데이터 내 라벨 분포:")
for i, (label, cnt) in enumerate(label_count_list, 1):
    print(f"{i:3d}. {label:15s}: {cnt}개")

# [2] 사용자 입력 (라벨, 데이터 수)
try:
    TOP_N = int(input("\n👉 변환할 라벨 개수(예: 6): ").strip())
except Exception:
    TOP_N = 6
print(f"✅ 변환할 라벨 개수: {TOP_N}")

try:
    MIN_SAMPLES = int(input("👉 라벨별 최대 변환 개수(예: 320): ").strip())
except Exception:
    MIN_SAMPLES = 320
print(f"✅ 라벨별 최대 변환 개수: {MIN_SAMPLES}")

TARGET_LABELS = [label for label, cnt in label_count_list[:TOP_N]]
PER_LABEL_LIMIT = MIN_SAMPLES

print(f"\n▶️ 변환 대상 라벨: {TARGET_LABELS}")

# [3] 변환 대상만 추출 (라벨/개수 카운팅)
converted_counts = {label: 0 for label in TARGET_LABELS}
task_list = []
for base_id, json_path, video_path in all_json_paths:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = data.get("potogrf", {}).get("fps", 30)
    sign_gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])
    for gloss in sign_gestures:
        gloss_id = str(gloss['gloss_id']).strip()
        if gloss_id not in TARGET_LABELS:
            continue
        # 저장 경로 및 중복 체크는 쓰레드별로 처리
        task_list.append({
            "base_id": base_id,
            "json_path": json_path,
            "video_path": video_path,
            "gloss": gloss,
            "fps": fps
        })

# [4] 병렬 변환 함수 정의
def process_one_gloss(task):
    base_id = task["base_id"]
    video_path = task["video_path"]
    gloss = task["gloss"]
    fps = task["fps"]
    gloss_id = str(gloss['gloss_id']).strip()
    gloss_id_clean = sanitize_filename(gloss_id)
    label_dir = os.path.join(output_dir, gloss_id_clean)
    os.makedirs(label_dir, exist_ok=True)
    out_name = f"{base_id}_{gloss_id_clean}.npy"
    out_path = os.path.join(label_dir, out_name)

    # 이미 존재하면 스킵
    if os.path.exists(out_path):
        return (gloss_id, False, out_path)

    start_sec = gloss['start']
    end_sec = gloss['end']
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    buffer_start = max(start_frame - BUFFER_FRAMES, 0)

    gloss_keypoints = []
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as holistic:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, buffer_start)
        for frame_idx in range(buffer_start, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            if frame_idx < start_frame:
                continue  # 🎯 buffer 프레임은 저장하지 않음

            lh = extract_landmarks(results.left_hand_landmarks, 3)
            rh = extract_landmarks(results.right_hand_landmarks, 3)
            pose = extract_landmarks(results.pose_landmarks, 4, idxs=POSE_INDEXES)
            keypoints = lh + rh + pose
            if len(keypoints) < expected_len:
                keypoints += [0.0] * (expected_len - len(keypoints))
            elif len(keypoints) > expected_len:
                keypoints = keypoints[:expected_len]
            gloss_keypoints.append(keypoints)
        cap.release()

    gloss_keypoints = np.array(gloss_keypoints)
    np.save(out_path, gloss_keypoints)
    return (gloss_id, True, out_path)

# [5] 병렬 변환 실행
final_counts = Counter()
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {}
    # 라벨별 변환 제한 걸기 위해 인덱스에서 개수 제한
    label_num = {label: 0 for label in TARGET_LABELS}
    for task in task_list:
        gloss_id = str(task["gloss"]['gloss_id']).strip()
        if label_num[gloss_id] >= PER_LABEL_LIMIT:
            continue
        label_num[gloss_id] += 1
        futures[executor.submit(process_one_gloss, task)] = gloss_id

    for future in as_completed(futures):
        gloss_id, saved, out_path = future.result()
        if saved:
            final_counts[gloss_id] += 1
            print(f"  ⬇️ 저장: {out_path}")
        else:
            print(f"  ⏩ 이미 존재, 건너뜀: {out_path}")

print("\n🎉 전체 변환 완료 (저장 기준)")
print("📊 새로 저장된 파일 개수:")
for label in TARGET_LABELS:
    print(f"  - {label}: {final_counts[label]}개 저장됨")