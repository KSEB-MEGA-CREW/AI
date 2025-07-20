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

# 🔹 MediaPipe 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7, min_tracking_confidence=0.7
)

POSE_SKIP_INDEXES = set(range(17, 33))
expected_len = 194      # 모델/데이터에 맞게 동일
BUFFER_SIZE = 10        # 현재 학습시 시퀀스 프레임
CONFIDENCE_THRESHOLD = 0.5

cap = cv2.VideoCapture(0)
frame_buffer = []
last_stable_label = None
stable_count = 0
output_sentence = []
STABLE_THRESHOLD = 3   # 연속 3프레임 예측 시 누적

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

print("🔁 실시간 수어 예측 시작 (q: 종료)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    display_frame = frame.copy()  # 화면에는 아무 텍스트도 안씀

    lh = extract_landmarks(results.left_hand_landmarks, 3)
    rh = extract_landmarks(results.right_hand_landmarks, 3)
    pose = extract_landmarks(results.pose_landmarks, 4, skip=POSE_SKIP_INDEXES)
    keypoints = lh + rh + pose

    # 입력 길이 맞추기 (패딩)
    if len(keypoints) < expected_len:
        keypoints += [0.0] * (expected_len - len(keypoints))
    elif len(keypoints) > expected_len:
        keypoints = keypoints[:expected_len]

    # 손 미인식/카메라 상태 체크
    zero_ratio = keypoints.count(0.0) / len(keypoints)
    if zero_ratio > 0.9:
        # 예측/누적 안함, skip만
        frame_buffer = []
        stable_count = 0
        last_stable_label = None
        cv2.imshow("실시간 수어 예측", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 버퍼 쌓기 (10프레임)
    frame_buffer.append(keypoints)
    if len(frame_buffer) > BUFFER_SIZE:
        frame_buffer.pop(0)

    predicted_label = None
    confidence = 0.0

    if len(frame_buffer) == BUFFER_SIZE:
        input_data = np.array(frame_buffer)
        # 정규화 (학습과 동일하게)
        max_abs = np.max(np.abs(input_data))
        if max_abs > 0:
            input_data = input_data / max_abs
        input_data = np.expand_dims(input_data, axis=0)
        prediction = model.predict(input_data, verbose=0)
        pred_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_label = label_map.get(pred_idx, "None") if confidence >= CONFIDENCE_THRESHOLD else "None"

        # 같은 단어가 STABLE_THRESHOLD번 연속 예측되면 누적
        if predicted_label != "None":
            if predicted_label == last_stable_label:
                stable_count += 1
            else:
                last_stable_label = predicted_label
                stable_count = 1
            if stable_count == STABLE_THRESHOLD:
                if len(output_sentence) == 0 or predicted_label != output_sentence[-1]:
                    output_sentence.append(predicted_label)
                    if len(output_sentence) > 10:
                        output_sentence = output_sentence[-10:]
                print(f"[RUN] 예측: {predicted_label}, 정확도: {confidence:.2f}, 누적 문장: {' '.join(output_sentence)}")
        else:
            last_stable_label = None
            stable_count = 0

    # 랜드마크 시각화 (옵션)
    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow("실시간 수어 예측", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()