import os
import json
from collections import Counter
import cv2
import mediapipe as mp
import numpy as np

### ===== [1] 데이터 경로 및 파라미터 =====
ROOT_DIR = "NIKL_Sign Language Parallel Corpus_2024_LI_CO"  # 최상위 폴더명(변경)
VIDEO_EXTS = [".mp4", ".avi", ".mov"]
OUTPUT_DIR = "output_npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

POSE_INDEXES = list(range(17))   # 하체 제외 (프로젝트 구조에 맞게 수정)
expected_len = 21*3 + 21*3 + 17*4

### ===== [2] gloss별 등장 횟수 집계 및 전체 json 경로 확보 =====
json_paths = []
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith(".json"):
            json_paths.append(os.path.join(root, file))

### ===== [3] npy 변환 =====
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

    for cam, cam_label in zip(['', 'L', 'R'], ['C', 'L', 'R']):
        video_file = os.path.join(parent_dir, f"{base_id}{cam}.mp4")
        if not os.path.exists(video_file):
            continue

        # --- 우선 모든 gloss마다 npy가 이미 있는지 먼저 체크 ---
        gloss_save_info = []
        for gloss in gestures:
            gloss_id = str(gloss.get("gloss_id", "")).replace('.npy', '').replace('.NPY', '')
            start_sec = gloss['start']
            end_sec = gloss['end']
            out_name = f"{base_id}_{gloss_id}_{cam_label}.npy"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            if os.path.exists(out_path):
                # 이미 npy 있으면 비디오 열지 않음!
                print(f"▶ {out_name} 이미 존재, 저장 건너뜀")
                continue
            gloss_save_info.append({
                "gloss_id": gloss_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "out_path": out_path
            })

        if not gloss_save_info:
            continue  # 저장해야 할 게 없으면, 비디오도 열지 않음

        # ---- [비디오 열기] 필요할 때만 ----
        cap = cv2.VideoCapture(video_file)
        all_keypoints = []
        with mp.solutions.holistic.Holistic(
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

        # ---- [npy 저장] ----
        for info in gloss_save_info:
            start_frame = int(round(info["start_sec"] * fps))
            end_frame = int(round(info["end_sec"] * fps))
            gloss_keypoints = all_keypoints[start_frame:end_frame+1]
            np.save(info["out_path"], gloss_keypoints)
            count_npy[info["gloss_id"]] += 1
            print(f"✅ {os.path.basename(info['out_path'])} 저장 완료 ({gloss_keypoints.shape})")

print(f"✅ npy 변환 완료! (gloss별 npy개수: {count_npy.most_common(10)})")