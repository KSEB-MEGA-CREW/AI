import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow.keras.models import load_model

# 🔹 모델 및 라벨 로딩
model = load_model("models/gesture_model.h5")
with open("models/label_map.json", "r", encoding="utf-8") as f:
    label_list = json.load(f)
label_map = {i: label for i, label in enumerate(label_list)}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

POSE_SKIP_INDEXES = set(range(17, 33))
expected_len = 194      # 모델/데이터에 맞게 동일
BUFFER_SIZE = 35        # 학습 시퀀스와 동일
CONFIDENCE_THRESHOLD = 0.5
STABLE_THRESHOLD = 5    # 연속 예측 프레임수(같은 단어 누적)

cap = cv2.VideoCapture(0)
frame_buffer = []
last_stable_label = None
stable_count = 0
output_sentence = []

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    display_frame = frame.copy()

    lh = extract_landmarks(results.left_hand_landmarks, 3)
    rh = extract_landmarks(results.right_hand_landmarks, 3)
    pose = extract_landmarks(results.pose_landmarks, 4, skip=POSE_SKIP_INDEXES)
    keypoints = lh + rh + pose

    # 입력 길이 맞추기
    if len(keypoints) < expected_len:
        keypoints += [0.0] * (expected_len - len(keypoints))
    elif len(keypoints) > expected_len:
        keypoints = keypoints[:expected_len]

    # 손 미인식/카메라 상태 체크
    zero_ratio = keypoints.count(0.0) / len(keypoints)
    if zero_ratio > 0.9:
        # (화면에 아무 메시지도 띄우지 않음)
        cv2.imshow("실시간 수어 예측", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_buffer = []
        stable_count = 0
        last_stable_label = None
        continue

    # 슬라이딩 윈도우 버퍼 (35프레임 유지)
    frame_buffer.append(keypoints)
    if len(frame_buffer) > BUFFER_SIZE:
        frame_buffer.pop(0)

    predicted_label = None
    confidence = 0.0

    if len(frame_buffer) == BUFFER_SIZE:
        # 예측 (정규화: 학습과 동일하게)
        input_data = np.array(frame_buffer)
        max_abs = np.max(np.abs(input_data))
        if max_abs > 0:
            input_data = input_data / max_abs
        input_data = np.expand_dims(input_data, axis=0)
        prediction = model.predict(input_data, verbose=0)
        pred_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_label = label_map.get(pred_idx, "None") if confidence >= CONFIDENCE_THRESHOLD else "None"

        # "같은 단어"가 연속 STABLE_THRESHOLD만큼 나오면 실제 단어로 인정
        if predicted_label != "None":
            if predicted_label == last_stable_label:
                stable_count += 1
            else:
                last_stable_label = predicted_label
                stable_count = 1
            if stable_count == STABLE_THRESHOLD:
                # 같은 단어가 문장에 누적되지 않게 마지막 단어와 비교
                if len(output_sentence) == 0 or predicted_label != output_sentence[-1]:
                    output_sentence.append(predicted_label)
                    if len(output_sentence) > 10:
                        output_sentence = output_sentence[-10:]
                print(f"[RUN] 예측: {predicted_label}, 정확도: {confidence:.2f}, 누적 문장: {' '.join(output_sentence)}")
        else:
            last_stable_label = None
            stable_count = 0

    # 랜드마크만 화면에 시각화(글씨 X)
    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow("실시간 수어 예측", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()