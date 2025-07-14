import cv2
import mediapipe as mp
import numpy as np
import os
import time

video_path = 'sample.mp4'
SAVE_DIR = "dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

POSE_INDEXES = list(range(17))  # 0~16만 사용
expected_len = 21*3 + 21*3 + 17*4  # lh(21*3) + rh(21*3) + pose(17*4)

mp_holistic = mp.solutions.holistic

def extract_landmarks(landmarks, dims, idxs=None):
    result = []
    if landmarks:
        # 원하는 인덱스만 추출 (idxs=None이면 전체)
        iter_landmarks = enumerate(landmarks.landmark) if idxs is None else ((i, landmarks.landmark[i]) for i in idxs)
        for i, lm in iter_landmarks:
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, 'visibility', 0.0))
            result.extend(coords)
    return result

label = input("수어 라벨을 입력하세요 (예: 사랑해요, 안녕하세요): ").strip()
keypoints_list = []

with mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as holistic:

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        lh = extract_landmarks(results.left_hand_landmarks, 3)
        rh = extract_landmarks(results.right_hand_landmarks, 3)
        # pose: 0~16번 keypoint만 사용, visibility 포함 4개 좌표
        pose = extract_landmarks(results.pose_landmarks, 4, idxs=POSE_INDEXES)

        if not lh and not rh:
            continue

        keypoints = lh + rh + pose
        if len(keypoints) < expected_len:
            keypoints += [0.0] * (expected_len - len(keypoints))
        elif len(keypoints) > expected_len:
            keypoints = keypoints[:expected_len]

        keypoints_list.append(keypoints)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"{frame_count} 프레임 처리 중...")

    cap.release()

keypoints_arr = np.array(keypoints_list)
filename = os.path.join(SAVE_DIR, f"{label}_{int(time.time())}.npy")
np.save(filename, keypoints_arr)
print(f"✅ 저장 완료: {filename} (shape={keypoints_arr.shape})")