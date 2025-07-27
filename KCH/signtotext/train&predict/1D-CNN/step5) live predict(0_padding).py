import cv2
import mediapipe as mp
import numpy as np
import json
import time
from tensorflow.keras.models import load_model

# ===== 모델 및 라벨 로딩 =====
MODEL_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\train&predict\1D-CNN\models\test\gesture_model.h5"
LABEL_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\train&predict\1D-CNN\models\test\label_map.json"

model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_list = json.load(f)
label_map = {i: label for i, label in enumerate(label_list)}

# ===== MediaPipe 초기화 =====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ===== 설정 =====
POSE_SKIP_INDEXES = set(range(17, 33))  # 하체 제외
EXPECTED_KEYPOINTS = 194
BUFFER_SIZE = 12
CONFIDENCE_THRESHOLD = 0.95
STABLE_THRESHOLD = 3
RESET_INTERVAL = 5.0
MAX_PADDING_RATIO = 0.44
MIN_VALID_FRAMES = 5
MIN_INTERVAL_BETWEEN_SAME_WORD = 1.0  # 초 단위

# ===== 상태 변수 =====
frame_buffer = []
last_stable_label = None
stable_count = 0
output_sentence = []
last_add_time = time.time()

cap = cv2.VideoCapture(0)

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

print("🔁 실시간 수어 예측 시작 (0패딩 기반, 'q'로 종료)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    display_frame = frame.copy()

    # 🔹 손이 모두 없는 경우 → None 출력
    if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
        frame_buffer.clear()
        stable_count = 0
        last_stable_label = None
        predicted_label = "none"
        confidence = 0.0
        print(f"[SKIP] 양손 없음 → None 출력")
        cv2.putText(display_frame, "none (0.00)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("실시간 수어 예측", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❎ 종료")
            break
        continue

    # 🔹 랜드마크 추출
    lh = extract_landmarks(results.left_hand_landmarks, dims=3)
    rh = extract_landmarks(results.right_hand_landmarks, dims=3)
    pose = extract_landmarks(results.pose_landmarks, dims=4, skip=POSE_SKIP_INDEXES)
    keypoints = lh + rh + pose

    # 🔹 keypoints 길이 조정 (0패딩)
    if len(keypoints) < EXPECTED_KEYPOINTS:
        keypoints += [0.0] * (EXPECTED_KEYPOINTS - len(keypoints))
    elif len(keypoints) > EXPECTED_KEYPOINTS:
        keypoints = keypoints[:EXPECTED_KEYPOINTS]

    # 🔹 예외 처리 (keypoint 거의 없음)
    zero_ratio = keypoints.count(0.0) / EXPECTED_KEYPOINTS
    if zero_ratio > MAX_PADDING_RATIO:
        frame_buffer.clear()
        stable_count = 0
        last_stable_label = None
        cv2.putText(display_frame, "none (0.00)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("실시간 수어 예측", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❎ 종료")
            break
        continue

    # 🔹 프레임 버퍼 쌓기
    frame_buffer.append(keypoints)
    if len(frame_buffer) > BUFFER_SIZE:
        frame_buffer.pop(0)

    predicted_label = "none"
    confidence = 0.0

    # 🔹 예측 조건 충족 시
    if len(frame_buffer) == BUFFER_SIZE:
        input_data = np.array(frame_buffer)
        max_abs = np.max(np.abs(input_data))
        if max_abs > 0:
            input_data = input_data / max_abs
        input_data = np.expand_dims(input_data, axis=0)

        prediction = model.predict(input_data, verbose=0)
        pred_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_label = label_map.get(pred_idx, "none") if confidence >= CONFIDENCE_THRESHOLD else "none"

        # 🔹 누적 문장에 추가 여부 판단
        if predicted_label != "none":
            if predicted_label == last_stable_label:
                stable_count += 1
            else:
                last_stable_label = predicted_label
                stable_count = 1

            if stable_count == STABLE_THRESHOLD:
                if not output_sentence or predicted_label != output_sentence[-1]:
                    if time.time() - last_add_time > MIN_INTERVAL_BETWEEN_SAME_WORD:
                        output_sentence.append(predicted_label)
                        last_add_time = time.time()
                        if len(output_sentence) > 10:
                            output_sentence = output_sentence[-10:]
                print(f"[RUN] 예측: {predicted_label}, 정확도: {confidence:.2f}, 문장: {' '.join(output_sentence)}")
        else:
            print(f"[NONE] 예측: none, 정확도: {confidence:.2f}")
            stable_count = 0
            last_stable_label = None

    # 🔹 시간 초과 시 누적 문장 초기화
    if output_sentence and (time.time() - last_add_time > RESET_INTERVAL):
        output_sentence = []
        print(f"[RESET] {RESET_INTERVAL}초 경과로 문장 초기화")

    # 🔹 시각화: 예측 결과 및 문장
    text_color = (0, 255, 0) if predicted_label != "none" else (0, 0, 255)
    cv2.putText(display_frame, f'{predicted_label} ({confidence:.2f})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
    cv2.putText(display_frame, ' '.join(output_sentence), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 🔹 랜드마크 시각화
    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow("실시간 수어 예측", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❎ 종료")
        break

cap.release()
cv2.destroyAllWindows()