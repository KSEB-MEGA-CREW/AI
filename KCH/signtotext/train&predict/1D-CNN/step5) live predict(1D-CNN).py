import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow.keras.models import load_model

# 모델/라벨 로딩
model = load_model("models/gesture_model.h5")
with open("models/label_map.json", "r", encoding="utf-8") as f:
    label_list = json.load(f)
label_map = {i: label for i, label in enumerate(label_list)}

# MediaPipe 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

POSE_SKIP_INDEXES = set(range(17, 33))
EXPECTED_LEN = 194
BUFFER_SIZE = 35
CONFIDENCE_THRESHOLD = 0.5
STABLE_THRESHOLD = 3   # 같은 단어 3번 연속이면 문장에 추가

cap = cv2.VideoCapture(0)
frame_buffer = []
output_sentence = []
stable_label = None
stable_count = 0

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

print("🔁 실시간 수어 예측 시작 (Q: 종료)")
print("예측 결과와 누적 문장은 아래 콘솔에 표시됩니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    display_frame = frame.copy()

    # 1프레임 랜드마크 추출
    lh = extract_landmarks(results.left_hand_landmarks, 3)
    rh = extract_landmarks(results.right_hand_landmarks, 3)
    pose = extract_landmarks(results.pose_landmarks, 4, skip=POSE_SKIP_INDEXES)
    keypoints = lh + rh + pose

    # 입력 feature 길이 맞춤
    if len(keypoints) < EXPECTED_LEN:
        keypoints += [0.0] * (EXPECTED_LEN - len(keypoints))
    elif len(keypoints) > EXPECTED_LEN:
        keypoints = keypoints[:EXPECTED_LEN]

    # 손/포즈 미인식(0이 90% 이상)일 경우 pass
    zero_ratio = keypoints.count(0.0) / len(keypoints)
    if zero_ratio > 0.9:
        frame_buffer = []
        stable_count = 0
        stable_label = None
        # 화면에는 랜드마크만 그리기(글씨 X)
        mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        cv2.imshow("실시간 수어 예측", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 35프레임 모으기
    frame_buffer.append(keypoints)
    if len(frame_buffer) < BUFFER_SIZE:
        # 화면에는 랜드마크만 그리기(글씨 X)
        mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        cv2.imshow("실시간 수어 예측", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # === 35프레임 쌓였을 때 예측 ===
    input_data = np.array(frame_buffer)
    max_abs = np.max(np.abs(input_data))
    if max_abs > 0:
        input_data = input_data / max_abs
    input_data = np.expand_dims(input_data, axis=0)

    prediction = model.predict(input_data, verbose=0)
    pred_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_label = label_map.get(pred_idx, "None") if confidence >= CONFIDENCE_THRESHOLD else "None"

    # 안정화: 같은 단어가 STABLE_THRESHOLD번 연속 나오면 문장에 추가
    if predicted_label != "None":
        if stable_label == predicted_label:
            stable_count += 1
        else:
            stable_label = predicted_label
            stable_count = 1
        if stable_count == STABLE_THRESHOLD:
            if len(output_sentence) == 0 or predicted_label != output_sentence[-1]:
                output_sentence.append(predicted_label)
                if len(output_sentence) > 10:
                    output_sentence = output_sentence[-10:]
            print(f"[RUN] 예측: {predicted_label}, 정확도: {confidence:.2f}, 누적 문장: {' '.join(output_sentence)}")
    else:
        stable_label = None
        stable_count = 0

    # 버퍼 초기화 (슬라이딩 X, 반복적 예측)
    frame_buffer = []

    # 화면에는 랜드마크만 그리기
    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    cv2.imshow("실시간 수어 예측", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()