import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

video_path = 'sample.mp4'
output_npy = 'output_keypoints.npy'

# 추출할 프레임 간격 (모든 프레임이면 1)
FRAME_INTERVAL = 1

keypoints_list = []

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False) as holistic:

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_INTERVAL != 0:
            frame_idx += 1
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        # 각 부위별 keypoint 추출 (없으면 0으로 패딩)
        def extract_landmarks(landmarks, num=21):  # 손은 21개, 포즈는 33개
            if landmarks:
                return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
            else:
                return np.zeros(num * 3)

        # 손, 포즈, 얼굴 등 원하는 것만 추출
        left_hand = extract_landmarks(results.left_hand_landmarks, 21)
        right_hand = extract_landmarks(results.right_hand_landmarks, 21)
        pose = extract_landmarks(results.pose_landmarks, 33)
        # 얼굴까지 쓰고 싶으면 아래 주석 해제
        # face = extract_landmarks(results.face_landmarks, 468)

        # 원하는 부위만 concat (예시: 양손+포즈)
        keypoints = np.concatenate([left_hand, right_hand, pose])  # (21*3 + 21*3 + 33*3 = 225)

        keypoints_list.append(keypoints)
        frame_idx += 1

    cap.release()

# numpy array로 변환 후 저장
keypoints_arr = np.array(keypoints_list)
np.save(output_npy, keypoints_arr)
print(f"Keypoints saved: {output_npy}, shape: {keypoints_arr.shape}")