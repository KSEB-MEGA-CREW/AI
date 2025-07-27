# 최적화된 얼굴+손+팔 포인트만 추출하는 .mp4 → .npy 변환 코드

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import re

def sanitize_filename(name):
    return re.sub(r'[:/\\?*<>|"]', '_', name)

root_folder = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\2024_LI_DC_0779230-0789690_1035"
output_dir = r"C:\KEB_bootcamp\project\AI\KCH\signtotext\output_npy\test"
os.makedirs(output_dir, exist_ok=True)

# 선택된 랜드마크 인덱스
SELECTED_FACE_INDEXES = [10,11,12,13,14,15,23,24,25,26,27,61,62,63,64,65,66,67,68,69,70,71,72]  # 눈썹, 코, 입 중심
POSE_INDEXES = [11,12,13,14,15,16]  # 어깨~손목
expected_len = 21*3 + 21*3 + len(SELECTED_FACE_INDEXES)*3 + len(POSE_INDEXES)*4

start_num = 240781060
stop_num = 240789690
step = +10

mp_holistic = mp.solutions.holistic

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

for num in range(start_num, stop_num, step):
    base_id = f"VXPAKOKS{num}"
    json_path = os.path.join(root_folder, f"{base_id}.json")
    video_path = os.path.join(root_folder, f"{base_id}.mp4")

    if not (os.path.exists(json_path) and os.path.exists(video_path)):
        print(f"  ⚠️ {base_id} 파일 없음")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = data.get("potogrf", {}).get("fps", 30)
    sign_gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])

    cap = cv2.VideoCapture(video_path)
    all_keypoints = []
    with mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            lh = extract_landmarks(results.left_hand_landmarks, 3)
            rh = extract_landmarks(results.right_hand_landmarks, 3)
            face = extract_landmarks(results.face_landmarks, 3, idxs=SELECTED_FACE_INDEXES)
            pose = extract_landmarks(results.pose_landmarks, 4, idxs=POSE_INDEXES)
            keypoints = lh + rh + face + pose
            if len(keypoints) < expected_len:
                keypoints += [0.0] * (expected_len - len(keypoints))
            elif len(keypoints) > expected_len:
                keypoints = keypoints[:expected_len]
            all_keypoints.append(keypoints)
    cap.release()
    all_keypoints = np.array(all_keypoints)

    for gloss in sign_gestures:
        gloss_id = str(gloss['gloss_id']).strip()
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

        if os.path.exists(out_path):
            print(f"  ⏩ 이미 존재, 건너뜀: {out_path}")
            continue

        np.save(out_path, gloss_keypoints)
        print(f"  ⬇️ 저장: {out_path} (shape={gloss_keypoints.shape})")

print("✅ 선택적 랜드마크 변환 완료")