import cv2
import mediapipe as mp
import numpy as np
import json
import os
import re

def sanitize_filename(name):
    return re.sub(r'[:\/\\?*<>|"]', '_', name)

# [경로 세팅]
folder = r"D:\NIKL_Sign Language Parallel Corpus_2024_FI_MR"
output_dir = r"C:\KEB_bootcamp\project\AI\KCH\signtotext\output_npy\금융입출금"
os.makedirs(output_dir, exist_ok=True)

POSE_INDEXES = list(range(17))
expected_len = 21*3 + 21*3 + 17*4

def extract_landmarks(landmarks, dims, idxs=None):
    result = []
    if landmarks:
        iter_landmarks = (
            enumerate(landmarks.landmark) if idxs is None
            else ((i, landmarks.landmark[i]) for i in idxs)
        )
        for i, lm in iter_landmarks:
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, 'visibility', 0.0))
            result.extend(coords)
    return result

# 1. 폴더 내 모든 json파일 탐색
json_files = []
for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith('.json'):
            json_files.append(os.path.join(root, file))

print(f"[INFO] json 파일 개수: {len(json_files)}개")

# 2. 각 json → 정면 mp4에 맞게 npy 변환
for json_file in json_files:
    base_id = os.path.splitext(os.path.basename(json_file))[0]
    video_dir = os.path.dirname(json_file)
    video_file = os.path.join(video_dir, f"{base_id}.mp4")  # 정면 카메라만

    if not os.path.exists(video_file):
        print(f"⚠️ 파일 없음: {video_file}")
        continue

    print(f"▶ 처리중: {base_id}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = data.get("potogrf", {}).get("fps", 30)
    sign_gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])

    cap = cv2.VideoCapture(video_file)
    all_keypoints = []
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.7,
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

    for gloss in sign_gestures:
        gloss_id = str(gloss['gloss_id'])
        gloss_id_clean = sanitize_filename(gloss_id.replace('.npy', '').replace('.NPY', ''))

        start_sec = gloss['start']
        end_sec = gloss['end']
        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        gloss_keypoints = all_keypoints[start_frame:end_frame+1]

        label_dir = os.path.join(output_dir, gloss_id_clean)
        os.makedirs(label_dir, exist_ok=True)
        out_name = f"{base_id}_{gloss_id_clean}.npy"
        out_path = os.path.join(label_dir, out_name)

        # [중복 방지] 이미 있으면 저장 안 함
        if os.path.exists(out_path):
            print(f"  ⏩ 이미 존재, 건너뜀: {out_path}")
            continue

        np.save(out_path, gloss_keypoints)
        print(f"  ⬇️ 저장: {out_path} (shape={gloss_keypoints.shape})")

print("✅ 변환 완료")