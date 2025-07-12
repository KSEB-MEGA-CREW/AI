import os
import json
from collections import Counter, defaultdict

### ===== [1] 데이터 경로 및 파라미터 =====
ROOT_DIR = "NIKL_Sign Language Parallel Corpus_2024_LI_CO"  # 최상위 폴더명(변경)
VIDEO_EXTS = [".mp4", ".avi", ".mov"]  # 비디오 확장자
OUTPUT_DIR = "output_npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_TOP_LABELS = 100   # [★] 상위 N개 라벨로만 추출

POSE_INDEXES = list(range(17))   # 하체 제외 (프로젝트 구조에 맞게 수정)
expected_len = 21*3 + 21*3 + 17*4

### ===== [2] gloss별 등장 횟수 집계 =====
gloss_counter = Counter()
json_paths = []

# 하위 폴더까지 모두 탐색
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith(".json"):
            json_path = os.path.join(root, file)
            json_paths.append(json_path)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])
            for gloss in gestures:
                gloss_id = str(gloss.get("gloss_id", "")).replace('.npy', '').replace('.NPY', '')
                gloss_counter[gloss_id] += 1

# 상위 N개 라벨 선정
top_gloss = [g for g, c in gloss_counter.most_common(N_TOP_LABELS)]
print(f"✅ 상위 {N_TOP_LABELS}개 gloss 선정: {top_gloss[:10]} ...")

### ===== [3] npy 변환(상위 라벨만) =====
import cv2
import mediapipe as mp
import numpy as np

def extract_landmarks(landmarks, dims, idxs=None):
    result = []
    if landmarks:
        iter_landmarks = enumerate(landmarks.landmark) if idxs is None else ((i, landmarks.landmark[i]) for i in idxs)
        for i, lm in iter_landmarks:
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, 'visibility', 0.0))
            result.extend(coords)
    return result

count_npy = Counter()

for json_path in json_paths:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = data["potogrf"]["fps"]
    gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])
    base_id = os.path.splitext(os.path.basename(json_path))[0]
    parent_dir = os.path.dirname(json_path)
    # [!] 모든 각도 cam 추출 지원
    for cam, cam_label in zip(['', 'L', 'R'], ['C', 'L', 'R']):
        video_file = os.path.join(parent_dir, f"{base_id}{cam}.mp4")
        if not os.path.exists(video_file):
            continue

        # 전체 프레임 랜드마크 추출
        cap = cv2.VideoCapture(video_file)
        all_keypoints = []
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                lh = extract_landmarks(results.left_hand_landmarks, 3)
                rh = extract_landmarks(results.right_hand_landmarks, 3)
                pose = extract_landmarks(results.pose_landmarks, 4, idxs=POSE_INDEXES)
                keypoints = lh + rh + pose
                if len(keypoints) < expected_len:
                    keypoints += [0.0] * (expected_len - len(keypoints))
                elif len(keypoints) > expected_len:
                    keypoints = keypoints[:expected_len]
                all_keypoints.append(keypoints)
            cap.release()
        all_keypoints = np.array(all_keypoints)

        # 각 gloss별로 npy 저장(상위 라벨만)
        for gloss in gestures:
            gloss_id = str(gloss.get("gloss_id", "")).replace('.npy', '').replace('.NPY', '')
            if gloss_id not in top_gloss:
                continue  # 상위 N개 라벨만 npy로 저장

            start_sec = gloss['start']
            end_sec = gloss['end']
            start_frame = int(round(start_sec * fps))
            end_frame = int(round(end_sec * fps))
            gloss_keypoints = all_keypoints[start_frame:end_frame+1]
            out_name = f"{base_id}_{gloss_id}_{cam_label}.npy"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            np.save(out_path, gloss_keypoints)
            count_npy[gloss_id] += 1

print(f"✅ npy 변환 완료! (gloss별 npy개수: {count_npy.most_common(10)})")