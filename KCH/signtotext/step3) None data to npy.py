import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# 데이터 저장 경로
SAVE_DIR = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\None"
os.makedirs(SAVE_DIR, exist_ok=True)

expected_len = 194   # 1프레임 keypoints feature 길이
SEQ_LEN = 10         # 수집할 프레임 수 (학습과 맞춰서)
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

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7, min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
print("[INFO] s키를 누르면 None 데이터 수집 시작 (프레임 10장 저장)")
print("[INFO] q키를 누르면 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    display_frame = frame.copy()
    cv2.putText(display_frame, "Press 's' to record None data, 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
    cv2.imshow("None Data Collection", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("수집 시작...")
        buffer = []
        collect_count = 0
        while collect_count < SEQ_LEN:
            ret, frame = cap.read()
            if not ret:
                continue
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # ① 손/포즈 인식 여부 확인
            if (
                results.left_hand_landmarks is None and
                results.right_hand_landmarks is None and
                results.pose_landmarks is None
            ):
                print("[SKIP] 손/포즈 인식 안 됨. 이 프레임은 None 데이터로 저장 X")
                continue  # skip

            lh = extract_landmarks(results.left_hand_landmarks, 3)
            rh = extract_landmarks(results.right_hand_landmarks, 3)
            pose = extract_landmarks(results.pose_landmarks, 4, skip=POSE_SKIP_INDEXES)
            keypoints = lh + rh + pose

            # 입력 길이 맞추기 (0패딩)
            if len(keypoints) < expected_len:
                keypoints += [0.0] * (expected_len - len(keypoints))
            elif len(keypoints) > expected_len:
                keypoints = keypoints[:expected_len]

            buffer.append(keypoints)
            collect_count += 1

            # 진행 상황 표시
            cv2.putText(frame, f"Collecting None ({collect_count}/{SEQ_LEN})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow("None Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # npy 파일 저장
        buffer_np = np.array(buffer)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"none_{now}.npy"
        np.save(os.path.join(SAVE_DIR, fname), buffer_np)
        print(f"[SAVE] None 데이터 저장: {fname}  shape={buffer_np.shape}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()