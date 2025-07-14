import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# 🔹 Conv1D + GAP 모델 및 라벨 로딩
model = load_model("models/gesture_model_conv1d_gap.h5")
with open("models/label_map_gap.json", "r", encoding="utf-8") as f:
    label_list = json.load(f)
label_map = {i: label for i, label in enumerate(label_list)}

# 🔹 한글 폰트 설정 (Windows 기준)
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
FONT = ImageFont.truetype(FONT_PATH, 28)
FONT_SMALL = ImageFont.truetype(FONT_PATH, 22)

# 🔹 MediaPipe 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

POSE_SKIP_INDEXES = set(range(17, 33))
expected_len = 194         # 모델 입력 크기
BUFFER_SIZE = 35           # Conv1D 모델 학습 시 프레임 수와 동일하게 유지
CONFIDENCE_THRESHOLD = 0.5

cap = cv2.VideoCapture(0)
frame_buffer = []
output_sentence = []
stable_label = None
stable_count = 0
STABLE_THRESHOLD = 5       # 연속 안정화 프레임 수

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

def draw_korean_text(frame, text, position=(10, 30), font=FONT, color=(0, 255, 0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

print("🔁 실시간 수어 예측 시작 (Q: 종료)")

predicted_label = ""
confidence = 0.0

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

    if len(keypoints) < expected_len:
        keypoints += [0.0] * (expected_len - len(keypoints))
    elif len(keypoints) > expected_len:
        keypoints = keypoints[:expected_len]

    zero_ratio = keypoints.count(0.0) / len(keypoints)
    if zero_ratio > 0.9:
        display_frame = draw_korean_text(display_frame, "손 인식 불가 (입력 무의미)", (10, 30), font=FONT, color=(0, 0, 255))
        cv2.imshow("실시간 수어 예측", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    frame_buffer.append(keypoints)
    if len(frame_buffer) > BUFFER_SIZE:
        frame_buffer.pop(0)

    predicted_label = None
    confidence = 0.0
    if len(frame_buffer) == BUFFER_SIZE:
        input_data = np.expand_dims(np.array(frame_buffer), axis=0)  # shape: (1, 35, 194)
        prediction = model.predict(input_data, verbose=0)
        pred_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_label = label_map.get(pred_idx, "None") if confidence >= CONFIDENCE_THRESHOLD else "None"

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
        else:
            stable_label = None
            stable_count = 0

    # 🔹 예측 출력
    if predicted_label not in [None, "None"]:
        label_text = f"예측: {predicted_label}"
        sub_text = f"(정확도: {confidence:.2f})"
        display_frame = draw_korean_text(display_frame, label_text, (10, 30), font=FONT, color=(0, 255, 0))
        display_frame = draw_korean_text(display_frame, sub_text, (10, 65), font=FONT_SMALL, color=(0, 180, 0))
        display_frame = draw_korean_text(display_frame, " ".join(output_sentence), (10, 100), font=FONT_SMALL, color=(255, 0, 0))
    else:
        display_frame = draw_korean_text(display_frame, "수어 인식 중...", (10, 30), font=FONT, color=(0, 0, 255))

    # 🔹 랜드마크 시각화
    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow("실시간 수어 예측", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()