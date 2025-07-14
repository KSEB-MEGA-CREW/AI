# save_none_npy.py
import cv2
import mediapipe as mp
import numpy as np
import os
import time

SAVE_DIR = "output_npy"
os.makedirs(SAVE_DIR, exist_ok=True)

LABEL = "none"
DURATION = 5         # 수집 시간 (초)
EXPECTED_LEN = 194   # keypoint 수
FPS = 10             # 초당 프레임 수 (너의 학습 기준이 35프레임이라면 대략 3.5초 정도면 충분)
FRAME_COUNT = DURATION * FPS

mp_holistic = mp.solutions.holistic
POSE_SKIP_INDEXES = set(range(17, 33))

def extract_landmarks(landmarks, dims, skip=None):
    result = []
    if landmarks:
        for i, lm in enumerate(landmarks.landmark):
            if skip and i in skip:
                continue
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, 'visibility', 0.0))
            result.extend(coords)
    return result

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

sample_id = 1
print(f"⏺️ '{LABEL}' 데이터 수집 시작 (Q: 종료, S: 저장 시작)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    cv2.putText(frame, f"Label: {LABEL} (S: 수집시작, Q: 종료)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("수어 (None) 수집", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        print("▶ 5초간 데이터 수집 시작...")
        sequence = []
        start_time = time.time()

        while len(sequence) < FRAME_COUNT:
            ret, frame = cap.read()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            lh = extract_landmarks(results.left_hand_landmarks, 3)
            rh = extract_landmarks(results.right_hand_landmarks, 3)
            pose = extract_landmarks(results.pose_landmarks, 4, skip=POSE_SKIP_INDEXES)
            keypoints = lh + rh + pose

            if len(keypoints) < EXPECTED_LEN:
                keypoints += [0.0] * (EXPECTED_LEN - len(keypoints))
            elif len(keypoints) > EXPECTED_LEN:
                keypoints = keypoints[:EXPECTED_LEN]

            sequence.append(keypoints)

            # 진행 표시
            cv2.putText(frame, f"Collecting frame {len(sequence)}/{FRAME_COUNT}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)
            cv2.imshow("수어 (None) 수집", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        npy_path = os.path.join(SAVE_DIR, f"none_{sample_id:03d}.npy")
        np.save(npy_path, np.array(sequence))
        print(f"✅ 저장 완료: {npy_path}")
        sample_id += 1

cap.release()
cv2.destroyAllWindows()