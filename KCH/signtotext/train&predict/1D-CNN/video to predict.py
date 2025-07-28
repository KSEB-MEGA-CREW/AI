import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow.keras.models import load_model
import time
import os

MODEL_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\train&predict\1D-CNN\models\일상_학교\gesture_model.h5"
LABEL_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\train&predict\1D-CNN\models\일상_학교\label_map.json"
VIDEO_SAVE_PATH = "recorded_sign_video.mp4"

model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_list = json.load(f)
label_map = {i: label for i, label in enumerate(label_list)}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

POSE_SKIP_INDEXES = set(range(17, 33))
expected_len = 194
BUFFER_SIZE = 12
CONFIDENCE_THRESHOLD = 0.99

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

def record_video(video_path):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20.0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    print("s: 녹화 시작, e: 녹화 종료")
    recording = False
    print("키 입력 대기 중...(s: 녹화 시작)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        key = cv2.waitKey(1) & 0xFF
        if not recording and key == ord('s'):
            print("녹화 시작!")
            recording = True
        if recording:
            out.write(frame)
            cv2.imshow("Recording...", frame)
        else:
            cv2.imshow("Webcam", frame)
        if recording and key == ord('e'):
            print("녹화 종료!")
            break
        if key == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return video_path if recording else None

def video_to_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )
    keypoints_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        lh = extract_landmarks(results.left_hand_landmarks, 3)
        rh = extract_landmarks(results.right_hand_landmarks, 3)
        pose = extract_landmarks(results.pose_landmarks, 4, skip=POSE_SKIP_INDEXES)
        keypoints = lh + rh + pose
        # 패딩/슬라이싱
        if len(keypoints) < expected_len:
            keypoints += [0.0] * (expected_len - len(keypoints))
        elif len(keypoints) > expected_len:
            keypoints = keypoints[:expected_len]
        zero_ratio = keypoints.count(0.0) / len(keypoints)
        if zero_ratio < 0.9:
            keypoints_list.append(keypoints)
    cap.release()
    holistic.close()
    return keypoints_list

def predict_words_from_landmarks(keypoints_list):
    output_sentence = []
    last_word = None
    stable_count = 0
    STABLE_THRESHOLD = 3
    i = 0
    while i + BUFFER_SIZE <= len(keypoints_list):
        buffer = keypoints_list[i:i+BUFFER_SIZE]
        input_data = np.array(buffer)
        max_abs = np.max(np.abs(input_data))
        if max_abs > 0:
            input_data = input_data / max_abs
        input_data = np.expand_dims(input_data, axis=0)
        prediction = model.predict(input_data, verbose=0)
        pred_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_label = label_map.get(pred_idx, "none") if confidence >= CONFIDENCE_THRESHOLD else "none"

        if predicted_label != "none":
            if predicted_label == last_word:
                stable_count += 1
            else:
                last_word = predicted_label
                stable_count = 1
            if stable_count == STABLE_THRESHOLD:
                if len(output_sentence) == 0 or predicted_label != output_sentence[-1]:
                    output_sentence.append(predicted_label)
        else:
            last_word = None
            stable_count = 0
        i += 1 # 1프레임씩 슬라이딩
    return output_sentence

if __name__ == "__main__":
    video_path = record_video(VIDEO_SAVE_PATH)
    if video_path:
        keypoints_list = video_to_landmarks(video_path)
        if len(keypoints_list) < BUFFER_SIZE:
            print("녹화된 수어 데이터가 너무 적습니다.")
        else:
            sentence = predict_words_from_landmarks(keypoints_list)
            print("\n[예측된 문장]:", " ".join(sentence))