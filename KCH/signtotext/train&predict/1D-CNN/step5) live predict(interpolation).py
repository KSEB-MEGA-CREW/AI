import cv2
import mediapipe as mp
import numpy as np
import json
import time
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d

# ===== 모델 및 라벨 로딩 =====
MODEL_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\train&predict\1D-CNN\models\보간\테스트\gesture_model.h5"
LABEL_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\train&predict\1D-CNN\models\보간\테스트\label_map.json"

model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_list = json.load(f)
label_map = {i: label for i, label in enumerate(label_list)}

# ===== MediaPipe 초기화 =====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ===== 설정 =====
EXPECTED_KEYPOINTS = 194  # pose 17개 x 4 + left 21 x 3 + right 21 x 3
REQUIRED_FRAMES = 12
CONFIDENCE_THRESHOLD = 0.98
STABLE_THRESHOLD = 3
RESET_INTERVAL = 5.0

frame_buffer = []
output_sentence = []
last_stable_label = None
stable_count = 0
last_add_time = time.time()

cap = cv2.VideoCapture(0)

# ===== 랜드마크 추출 함수 =====
def extract_landmarks(landmarks, dims=3, skip_idxs=None):
    result = []
    if landmarks:
        for i, lm in enumerate(landmarks.landmark):
            if skip_idxs and i in skip_idxs:
                continue
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, 'visibility', 0.0))
            result.extend(coords[:dims])
    return result

# ===== 보간 함수 =====
def interpolate_sequence(sequence, target_len=REQUIRED_FRAMES):
    seq = np.array(sequence)
    current_len = seq.shape[0]
    if current_len == target_len:
        return seq
    x_old = np.linspace(0, 1, num=current_len)
    x_new = np.linspace(0, 1, num=target_len)
    interpolated = interp1d(x_old, seq, axis=0, kind='linear', fill_value="extrapolate")(x_new)
    return interpolated

print("🔁 실시간 수어 예측 시작 (보간 기반 + visibility 포함, q: 종료)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    display_frame = frame.copy()

    # 🔹 keypoint 추출 (pose: 4D, hand: 3D)
    pose = extract_landmarks(results.pose_landmarks, dims=4, skip_idxs=range(17, 33))  # 17개 x 4D = 68
    lh = extract_landmarks(results.left_hand_landmarks, dims=3)  # 21 x 3D = 63
    rh = extract_landmarks(results.right_hand_landmarks, dims=3)  # 21 x 3D = 63
    keypoints = pose + lh + rh  # 총합: 68 + 63 + 63 = 194

    # 🔹 예외 처리
    if len(keypoints) != EXPECTED_KEYPOINTS:
        print(f"⚠️ 예외 keypoint 수: {len(keypoints)} (예상: {EXPECTED_KEYPOINTS})")
        cv2.imshow("실시간 수어 예측 (보간)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    frame_buffer.append(keypoints)
    if len(frame_buffer) > 30:
        frame_buffer.pop(0)

    predicted_label = None
    confidence = 0.0

    if len(frame_buffer) >= 7:
        input_seq = interpolate_sequence(frame_buffer, REQUIRED_FRAMES)
        max_abs = np.max(np.abs(input_seq))
        if max_abs > 0:
            input_seq = input_seq / max_abs
        input_data = np.expand_dims(input_seq, axis=0)

        prediction = model.predict(input_data, verbose=0)
        pred_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_label = label_map.get(pred_idx, "none") if confidence >= CONFIDENCE_THRESHOLD else "none"

        if predicted_label != "none":
            if predicted_label == last_stable_label:
                stable_count += 1
            else:
                last_stable_label = predicted_label
                stable_count = 1

            if stable_count == STABLE_THRESHOLD:
                if len(output_sentence) == 0 or predicted_label != output_sentence[-1]:
                    output_sentence.append(predicted_label)
                    last_add_time = time.time()
                    if len(output_sentence) > 10:
                        output_sentence = output_sentence[-10:]
                print(f"[RUN] 예측: {predicted_label}, 정확도: {confidence:.2f}, 누적 문장: {' '.join(output_sentence)}")
        else:
            print(f"[NONE] 예측: none, 정확도: {confidence:.2f}")
            last_stable_label = None
            stable_count = 0

    if len(output_sentence) > 0 and (time.time() - last_add_time > RESET_INTERVAL):
        output_sentence = []
        print(f"[RESET] {RESET_INTERVAL}초 경과로 누적 문장 초기화")

    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow("실시간 수어 예측 (보간 + visibility)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()