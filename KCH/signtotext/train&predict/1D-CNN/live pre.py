"""
Live prediction script for the 1D-CNN hand gesture recognizer (with pose).

This version uses MediaPipe Holistic to extract both hand and upper-body pose landmarks
for improved gesture recognition accuracy in real-time prediction.
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
from collections import deque
from typing import Deque, List, Tuple
from PIL import ImageFont, ImageDraw, Image

REQUIRED_FRAMES: int = 12
EXPECTED_LEN: int = 194
MODEL_PATH: str = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\train&predict\1D-CNN\models\일상_학교(4)\gesture_model.h5"
LABEL_MAP_PATH: str = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\train&predict\1D-CNN\models\일상_학교(4)\label_map.json"
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
FONT_SIZE = 30
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# 사용되는 포즈 인덱스는 0~16번 (상체 중심)
POSE_INDEXES = list(range(17))


def extract_features(results) -> np.ndarray:
    hands_array = np.zeros((2, 21, 3), dtype=np.float32)
    if results.left_hand_landmarks:
        for j, lm in enumerate(results.left_hand_landmarks.landmark):
            hands_array[0, j] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for j, lm in enumerate(results.right_hand_landmarks.landmark):
            hands_array[1, j] = [lm.x, lm.y, lm.z]

    pose_array = np.zeros((len(POSE_INDEXES), 4), dtype=np.float32)
    if results.pose_landmarks:
        for i in POSE_INDEXES:
            lm = results.pose_landmarks.landmark[i]
            pose_array[i] = [lm.x, lm.y, lm.z, lm.visibility]

    # 평탄화 + 결합
    flat_hand = hands_array.reshape(-1)
    flat_pose = pose_array.reshape(-1)
    combined = np.concatenate([flat_hand, flat_pose])

    # 패딩 또는 자르기
    if combined.size < EXPECTED_LEN:
        combined = np.pad(combined, (0, EXPECTED_LEN - combined.size))
    elif combined.size > EXPECTED_LEN:
        combined = combined[:EXPECTED_LEN]
    return combined


def predict_from_queue(sequence_queue: Deque[np.ndarray], model: tf.keras.Model,
                       labels: List[str]) -> Tuple[str, float]:
    sequence = np.stack(sequence_queue, axis=0)
    max_abs = np.max(np.abs(sequence))
    if max_abs > 0:
        sequence = sequence / max_abs
    input_data = np.expand_dims(sequence, axis=0)
    preds = model.predict(input_data, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    pred_label = labels[pred_idx]
    return pred_label, confidence


def draw_text_korean(img, text, position, font, font_size=30, color=(0, 255, 0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)


def main() -> None:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    sequence_queue: Deque[np.ndarray] = deque(maxlen=REQUIRED_FRAMES)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting live prediction with pose. Press 'q' to quit.")
    current_prediction = ""
    current_confidence = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed. Exiting.")
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True

            features = extract_features(results)
            sequence_queue.append(features)

            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            if len(sequence_queue) == REQUIRED_FRAMES:
                pred_label, confidence = predict_from_queue(sequence_queue, model, labels)
                current_prediction = pred_label
                current_confidence = confidence

            if current_prediction:
                text = f"{current_prediction} ({current_confidence:.2f})"
                frame = draw_text_korean(frame, text, (10, 30), font)

            cv2.imshow('Live Prediction with Pose (Press q to exit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()