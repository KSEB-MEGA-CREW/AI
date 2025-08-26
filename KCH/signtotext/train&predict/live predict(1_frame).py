import cv2
import mediapipe as mp
import numpy as np
import json
import time
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image  # 한글 출력용
from collections import Counter, defaultdict

# ===== 한글 텍스트 출력 함수 =====
def draw_text_korean(frame, text, position, font_size=30, color=(255,255,255)):
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 한글 폰트 경로
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ===== 모델 및 라벨 로딩 =====
MODEL_PATH = r"C:\Users\cksgu\Desktop\김찬혁\2025\KSEB\프로젝트\v5_cnn\gesture_model.h5"
LABEL_PATH = r"C:\Users\cksgu\Desktop\김찬혁\2025\KSEB\프로젝트\v5_cnn\label_map.json"

model = load_model(MODEL_PATH, compile=False)
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
BUFFER_SIZE = 10  # ← train 프레임 수와 일치
CONFIDENCE_THRESHOLD = 0.96
STABLE_THRESHOLD = 2
RESET_INTERVAL = 5.0
MAX_PADDING_RATIO = 0.5
MIN_VALID_FRAMES = 6
MIN_INTERVAL_BETWEEN_SAME_WORD = 1.0  # 초 단위

# ===== 상태 변수 =====
frame_buffer = []          # 실시간 예측용 순환 버퍼
last_stable_label = None
stable_count = 0
output_sentence = []
last_add_time = time.time()

# 녹화 관련
recording = False
record_buffer = []         # 녹화 구간의 프레임(키포인트) 저장
last_record_result = None  # {'text': str, 'expire': timestamp}

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

def preprocess_keypoints(lh, rh, pose):
    keypoints = lh + rh + pose
    # 0패딩/자르기
    if len(keypoints) < EXPECTED_KEYPOINTS:
        keypoints += [0.0] * (EXPECTED_KEYPOINTS - len(keypoints))
    elif len(keypoints) > EXPECTED_KEYPOINTS:
        keypoints = keypoints[:EXPECTED_KEYPOINTS]
    return keypoints

def is_valid_frame(keypoints):
    zero_ratio = keypoints.count(0.0) / EXPECTED_KEYPOINTS
    return zero_ratio <= MAX_PADDING_RATIO

def windowed_predictions(frames, buffer_size, threshold):
    """
    frames: [ [EXPECTED_KEYPOINTS], ... ]  (길이 가변)
    buffer_size: 윈도우 크기 (모델 입력 프레임 수)
    return:
      preds: [("라벨" 또는 "none", conf), ...]  # stride=1
    """
    preds = []
    n = len(frames)
    if n == 0:
        return preds

    # 길이가 짧으면 패딩해서 1회 예측
    if n < buffer_size:
        window = frames + [[0.0]*EXPECTED_KEYPOINTS for _ in range(buffer_size - n)]
        window = np.array(window, dtype=np.float32)
        max_abs = np.max(np.abs(window))
        if max_abs > 0:
            window = window / max_abs
        window = np.expand_dims(window, axis=0)
        p = model.predict(window, verbose=0)
        idx = int(np.argmax(p))
        conf = float(np.max(p))
        label = label_map.get(idx, "none") if conf >= threshold else "none"
        preds.append((label, conf))
        return preds

    # 슬라이딩 윈도우 stride=1
    for s in range(0, n - buffer_size + 1):
        window = np.array(frames[s:s + buffer_size], dtype=np.float32)
        max_abs = np.max(np.abs(window))
        if max_abs > 0:
            window = window / max_abs
        window = np.expand_dims(window, axis=0)
        p = model.predict(window, verbose=0)
        idx = int(np.argmax(p))
        conf = float(np.max(p))
        label = label_map.get(idx, "none") if conf >= threshold else "none"
        preds.append((label, conf))
    return preds

def summarize_predictions(preds):
    """
    preds: [ (label, conf), ... ]
    return:
      majority_label, majority_conf_mean,
      seq_compact (연속 동일 라벨 압축, 'none' 제거) as list[str],
      counts_dict (라벨별 개수, 'none' 포함)
    """
    if not preds:
        return "none", 0.0, [], {}

    counts = Counter([lab for lab, _ in preds])
    # 최다 라벨(가능하면 'none' 제외) 선택
    if len(counts) > 1 and 'none' in counts:
        # none 제외한 것 중 최다
        non_none_counts = {k: v for k, v in counts.items() if k != 'none'}
        if non_none_counts:
            majority_label = max(non_none_counts, key=non_none_counts.get)
        else:
            majority_label = 'none'
    else:
        majority_label = counts.most_common(1)[0][0]

    # 해당 라벨의 평균 confidence
    confs = [conf for lab, conf in preds if lab == majority_label]
    majority_conf_mean = float(np.mean(confs)) if confs else 0.0

    # 연속 동일 라벨 압축 (none 제거)
    seq_compact = []
    last = None
    for lab, _ in preds:
        if lab == 'none':
            last = lab
            continue
        if lab != last:
            seq_compact.append(lab)
        last = lab

    return majority_label, majority_conf_mean, seq_compact, dict(counts)

print("🔁 실시간 수어 예측 시작 (0패딩 기반, 'q' 종료 / 'r' 녹화 토글 / 'c' 녹화 취소)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    display_frame = frame.copy()

    # 🔹 랜드마크 추출
    lh = extract_landmarks(results.left_hand_landmarks, dims=3)
    rh = extract_landmarks(results.right_hand_landmarks, dims=3)
    pose = extract_landmarks(results.pose_landmarks, dims=4, skip=POSE_SKIP_INDEXES)
    keypoints = preprocess_keypoints(lh, rh, pose)
    valid = is_valid_frame(keypoints)

    # 🔹 실시간 예측 로직 (기존 유지)
    predicted_label = "none"
    confidence = 0.0

    # 손이 모두 없는 경우: 실시간 버퍼/상태 리셋
    if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
        frame_buffer.clear()
        stable_count = 0
        last_stable_label = None
        predicted_label = "none"
        confidence = 0.0
        display_frame = draw_text_korean(display_frame, "없음 (0.00)", (10, 30), 32, (0, 0, 255))
    else:
        # 유효 프레임만 실시간 버퍼에 사용
        if valid:
            frame_buffer.append(keypoints)
            if len(frame_buffer) > BUFFER_SIZE:
                frame_buffer.pop(0)

        if len(frame_buffer) == BUFFER_SIZE:
            input_data = np.array(frame_buffer, dtype=np.float32)
            max_abs = np.max(np.abs(input_data))
            if max_abs > 0:
                input_data = input_data / max_abs
            input_data = np.expand_dims(input_data, axis=0)

            prediction = model.predict(input_data, verbose=0)
            pred_idx = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            predicted_label = label_map.get(pred_idx, "none") if confidence >= CONFIDENCE_THRESHOLD else "none"

            # 문장 누적
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
            else:
                stable_count = 0
                last_stable_label = None

        # 시간 초과 시 문장 초기화
        if output_sentence and (time.time() - last_add_time > RESET_INTERVAL):
            output_sentence = []

        # 시각화
        label_text = "없음" if predicted_label == "none" else predicted_label
        display_frame = draw_text_korean(display_frame, f'{label_text} ({confidence:.2f})', (10, 30),
                                         font_size=32, color=(0,255,0) if predicted_label != "none" else (0,0,255))
        display_frame = draw_text_korean(display_frame, ' '.join(output_sentence), (10, 70),
                                         font_size=28, color=(255,255,255))

    # 🔹 녹화 중이면 유효 프레임만 저장
    if recording and valid:
        record_buffer.append(keypoints)

    # 🔹 랜드마크 시각화
    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # 🔹 녹화 상태 표시
    if recording:
        display_frame = draw_text_korean(display_frame, f'● REC  프레임: {len(record_buffer)}',
                                         (10, 110), font_size=28, color=(0,0,255))

    # 🔹 최근 녹화 결과 오버레이 (3초 표시)
    if last_record_result and time.time() < last_record_result['expire']:
        display_frame = draw_text_korean(display_frame, last_record_result['text'],
                                         (10, 150), font_size=26, color=(255, 215, 0))

    cv2.imshow("실시간 수어 예측 + 녹화 테스트", display_frame)

    # 🔹 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("❎ 종료")
        break
    elif key == ord('r'):
        # 녹화 토글
        recording = not recording
        if recording:
            record_buffer = []
            print("⏺ 녹화 시작")
            last_record_result = None
        else:
            # 녹화 종료 → 분석
            print(f"⏹ 녹화 중지. 수집 프레임: {len(record_buffer)}")
            preds = windowed_predictions(record_buffer, BUFFER_SIZE, CONFIDENCE_THRESHOLD)
            majority_label, majority_conf, seq_compact, counts = summarize_predictions(preds)

            # 콘솔 출력
            print(f"[녹화 결과] 프레임={len(record_buffer)} 윈도우={len(preds)}")
            print(f" - 최다 라벨: {majority_label} (avg conf: {majority_conf:.2f})")
            print(f" - 연속 시퀀스: {' '.join(seq_compact) if seq_compact else '(없음)'}")
            print(f" - 라벨 카운트: {counts}")

            # 화면 오버레이용 메시지 (3초 노출)
            overlay_text = f"[REC 결과] {majority_label} ({majority_conf:.2f}) | 시퀀스: {' '.join(seq_compact) if seq_compact else '(없음)'}"
            last_record_result = {'text': overlay_text, 'expire': time.time() + 3.0}
    elif key == ord('c'):
        # 녹화 취소
        if recording or record_buffer:
            recording = False
            record_buffer = []
            last_record_result = {'text': "[REC] 취소됨", 'expire': time.time() + 2.0}
            print("🗑 녹화 취소 및 버퍼 삭제")

cap.release()
holistic.close()
cv2.destroyAllWindows()