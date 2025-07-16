import cv2
import mediapipe as mp
import numpy as np
import json
import os
import re
from collections import Counter

def sanitize_filename(name):
    return re.sub(r'[:\/\\?*<>|"]', '_', name)

folder = "2024_LI_DC_0779230-0789690_1035"
output_dir = "output_npy"
os.makedirs(output_dir, exist_ok=True)

POSE_INDEXES = list(range(17))
expected_len = 21*3 + 21*3 + 17*4

start_num = 240779260
file_count = 1

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

# 1. 하위 폴더까지 모든 json파일 탐색
json_files = []
for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith('.json'):
            json_files.append(os.path.join(root, file))

# 2. gloss_id 카운트
gloss_counter = Counter()
for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"⚠️ JSON 오류: {json_file} ({e})")
            continue
        sign_gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])
        for gloss in sign_gestures:
            gloss_id = str(gloss['gloss_id'])
            gloss_id_clean = sanitize_filename(gloss_id.replace('.npy', '').replace('.NPY', ''))
            gloss_counter[gloss_id_clean] += 1

# 3. 상위 100개 gloss만 선별
top_glosses = set([g for g, _ in gloss_counter.most_common(100)])
print(f"상위 100개 gloss 예시: {list(top_glosses)[:10]} ...")

# 4. 실제 변환(정면 cam만)
for i in range(file_count):
    base_id = f"VXPAKOKS{start_num + 10*i}"
    cam = ''
    # json 파일이 어느 폴더에 있든 찾아서 연결
    json_file = next((f for f in json_files if os.path.basename(f) == f"{base_id}.json"), None)
    video_file = None

    # json 파일이 있다면, 같은 위치의 mp4를 찾음 (정면 cam)
    if json_file:
        video_dir = os.path.dirname(json_file)
        video_file_candidate = os.path.join(video_dir, f"{base_id}{cam}.mp4")
        if os.path.exists(video_file_candidate):
            video_file = video_file_candidate

    if not (json_file and video_file):
        print(f"⚠️ 파일 없음: {json_file} 또는 {video_file}")
        continue

    print(f"▶ 처리중: {base_id}{cam}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = data["potogrf"]["fps"]
    sign_gestures = data["sign_script"]["sign_gestures_strong"]

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

        # 상위 100개 gloss만 npy로 저장
        if gloss_id_clean not in top_glosses:
            continue

        start_sec = gloss['start']
        end_sec = gloss['end']
        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        gloss_keypoints = all_keypoints[start_frame:end_frame+1]

        label_dir = os.path.join(output_dir, gloss_id_clean)
        os.makedirs(label_dir, exist_ok=True)
        # 여기서 cam_label 없이 파일명 지정!
        out_name = f"{base_id}_{gloss_id_clean}.npy"
        out_path = os.path.join(label_dir, out_name)
        np.save(out_path, gloss_keypoints)
        print(f"  ⬇️ 저장: {out_path} (shape={gloss_keypoints.shape})")