import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ===== 설정 =====
SAVE_DIR = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\My data\dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

POSE_SKIP_INDEXES = set(range(17, 33))  # 하체 제외
EXPECTED_LEN = 194
SAVE_FRAMES = 8
LABEL_NAME = "ㅕ"
STABLE_SKIP_FRAMES = 4  # 앞의 불안정한 프레임 개수

# ===== MediaPipe 초기화 =====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def extract_landmarks(landmarks, dims=3, skip=None):
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

cap = cv2.VideoCapture(0)
print("▶ 's' 키를 눌러 수집 시작, 'q' 키로 종료")

state = "idle"  # 'idle', 'waiting', 'collecting'
start_time = None
data_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    display_frame = frame.copy()

    # 랜드마크 시각화
    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    current_time = time.time()

    # 상태 처리
    if state == "waiting":
        elapsed = current_time - start_time
        cv2.putText(display_frame, f"⏱️ 대기 중: {elapsed:.1f}s", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if elapsed >= 1.0:
            state = "collecting"
            print("📸 수집 시작!")

    elif state == "collecting":
        lh = extract_landmarks(results.left_hand_landmarks)
        rh = extract_landmarks(results.right_hand_landmarks)
        pose = extract_landmarks(results.pose_landmarks, dims=4, skip=POSE_SKIP_INDEXES)

        keypoints = lh + rh + pose
        if len(keypoints) < EXPECTED_LEN:
            keypoints += [0.0] * (EXPECTED_LEN - len(keypoints))
        elif len(keypoints) > EXPECTED_LEN:
            keypoints = keypoints[:EXPECTED_LEN]

        data_buffer.append(keypoints)

        cv2.putText(display_frame, f"수집 중: {len(data_buffer)}/{SAVE_FRAMES + STABLE_SKIP_FRAMES}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if len(data_buffer) >= SAVE_FRAMES + STABLE_SKIP_FRAMES:
            output_array = np.array(data_buffer[STABLE_SKIP_FRAMES:])
            filename = f"{int(time.time())}_{LABEL_NAME}.npy"
            save_path = os.path.join(SAVE_DIR, filename)
            np.save(save_path, output_array)
            print(f"✅ 저장 완료: {save_path}")
            data_buffer = []
            state = "idle"
            start_time = None

    elif state == "idle":
        cv2.putText(display_frame, "▶ 's' 누르면 수집 시작", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and state == "idle":
        state = "waiting"
        start_time = time.time()
        data_buffer = []
        print("🟢 1초 대기 후 수집을 시작합니다...")
    elif key == ord('q'):
        print("❎ 종료")
        break

    cv2.imshow("실시간 수어 데이터 수집", display_frame)

cap.release()
cv2.destroyAllWindows()