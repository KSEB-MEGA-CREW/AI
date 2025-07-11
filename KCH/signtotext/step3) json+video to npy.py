import cv2
import mediapipe as mp
import numpy as np
import json
import os

# 파일 경로 세팅
json_path = "VXPAKOKS240779230.json"
video_path = "VXPAKOKS240779230.mp4"
output_dir = "output_npy"
os.makedirs(output_dir, exist_ok=True)

# MediaPipe 세팅
POSE_INDEXES = list(range(17))   # 0~16번만 사용
expected_len = 21*3 + 21*3 + 17*4  # lh + rh + pose

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

# 1. JSON 파싱
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

fps = data["potogrf"]["fps"]
sign_gestures = data["sign_script"]["sign_gestures_strong"]

# 2. 전체 영상 MediaPipe 분석
cap = cv2.VideoCapture(video_path)

all_keypoints = []
frame_idx = 0
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
        # 길이 맞추기
        keypoints = lh + rh + pose
        if len(keypoints) < expected_len:
            keypoints += [0.0] * (expected_len - len(keypoints))
        elif len(keypoints) > expected_len:
            keypoints = keypoints[:expected_len]
        all_keypoints.append(keypoints)
        frame_idx += 1

    cap.release()

all_keypoints = np.array(all_keypoints)
print(f"전체 프레임 keypoints shape: {all_keypoints.shape}")

# 3. 각 gloss_id별로 잘라서 npy로 저장
for gloss in sign_gestures:
    start_sec = gloss['start']
    end_sec = gloss['end']
    gloss_id = gloss['gloss_id']

    # 구간별 frame index 계산
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    gloss_keypoints = all_keypoints[start_frame:end_frame+1]

    out_name = f"{data['id']}_{gloss_id}.npy"
    out_path = os.path.join(output_dir, out_name)
    np.save(out_path, gloss_keypoints)
    print(f"저장: {out_path} (shape={gloss_keypoints.shape})")