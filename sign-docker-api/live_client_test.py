# live_client_test.py
# - 카메라로부터 프레임 -> MediaPipe로 keypoints 추출 -> REST(POST /predict/frame)로 전송
# - 서버 응답(label, confidence)을 받아서 화면에 한글로 표시 + 문장 누적 로직

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import uuid
import requests
from PIL import ImageFont, ImageDraw, Image  # 한글 출력용

# ====================== 설정 ======================
API_URL = "http://localhost:8000/predict/frame"  # 도커 API 엔드포인트
SESSION_ID = str(uuid.uuid4())                   # 세션 고유 ID (기기/브라우저별로 고정 권장)
TIMEOUT = 3                                      # 요청 타임아웃(초)

# 모델/전처리 스펙(서버와 반드시 동일)
POSE_SKIP_INDEXES = set(range(17, 33))  # 하체 제외
EXPECTED_KEYPOINTS = 194
WINDOW = 10                   # 서버 WINDOW와 동일
CONFIDENCE_THRESHOLD = 0.4    # 서버 CONF_THRESHOLD와 동일(서버도 최종적용함)
MAX_PADDING_RATIO = 0.5       # 유효성 필터(클라 측)
RESET_INTERVAL = 5.0          # 문장 초기화 간격(초)
STABLE_THRESHOLD = 3          # 같은 라벨 연속 감지 시 문장 누적
MIN_INTERVAL_BETWEEN_SAME_WORD = 1.0  # 같은 단어 연속 방지 간격(초)

# ================== 한글 텍스트 출력 =================
def draw_text_korean(frame, text, position, font_size=30, color=(255,255,255)):
    # Windows 폰트 경로
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ================== MediaPipe 초기화 =================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ================== 상태 변수 ======================
last_label = None
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

print("🔁 실시간 수어 예측 (REST 개별 전송 / q로 종료)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    display_frame = frame.copy()

    # 양손 모두 없으면 skip
    if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
        stable_count = 0
        last_label = None
        label_text = "없음"
        confidence = 0.0
        display_frame = draw_text_korean(display_frame, f"{label_text} ({confidence:.2f})", (10, 30), 32, (0, 0, 255))
        display_frame = draw_text_korean(display_frame, ' '.join(output_sentence), (10, 70), 28, (255,255,255))
        cv2.imshow("실시간 수어 예측(REST)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 랜드마크 추출 (좌/우손 + 상체 포즈)
    lh = extract_landmarks(results.left_hand_landmarks, dims=3)
    rh = extract_landmarks(results.right_hand_landmarks, dims=3)
    pose = extract_landmarks(results.pose_landmarks, dims=4, skip=POSE_SKIP_INDEXES)
    keypoints = lh + rh + pose

    # 0패딩/자르기
    if len(keypoints) < EXPECTED_KEYPOINTS:
        keypoints += [0.0] * (EXPECTED_KEYPOINTS - len(keypoints))
    elif len(keypoints) > EXPECTED_KEYPOINTS:
        keypoints = keypoints[:EXPECTED_KEYPOINTS]

    # 유효성 필터(클라이언트 측)
    zero_ratio = keypoints.count(0.0) / EXPECTED_KEYPOINTS
    if zero_ratio > MAX_PADDING_RATIO:
        stable_count = 0
        last_label = None
        label_text = "무효"
        confidence = 0.0
        display_frame = draw_text_korean(display_frame, f"{label_text} ({confidence:.2f})", (10, 30), 32, (0, 0, 255))
        display_frame = draw_text_korean(display_frame, ' '.join(output_sentence), (10, 70), 28, (255,255,255))
        cv2.imshow("실시간 수어 예측(REST)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ====== 서버로 프레임 1개 전송 (REST) ======
    try:
        payload = {
            "session_id": SESSION_ID,
            "keypoints": keypoints  # 서버는 길이 194 벡터 1개를 받음
        }
        r = requests.post(API_URL, json=payload, timeout=TIMEOUT)
        if r.status_code != 200:
            # 서버가 4xx/5xx를 던지면 로그만 찍고 다음 프레임 진행
            print(f"[HTTP {r.status_code}] {r.text}")
            label_text = "에러"
            confidence = 0.0
        else:
            resp = r.json()
            # 수집 중 상태
            if resp.get("status") == "collecting":
                collected = resp.get("collected", 0)
                label_text = f"수집중 {collected}/{WINDOW}"
                confidence = 0.0
            else:
                # 예측 완료 응답
                predicted_label = resp.get("label", "None")
                confidence = float(resp.get("confidence", 0.0))
                label_text = "없음" if predicted_label in ["None", "none"] else predicted_label

                # 문장 누적 (클라 측 안정화 로직)
                if predicted_label not in ["None", "none"]:
                    if predicted_label == last_label:
                        stable_count += 1
                    else:
                        last_label = predicted_label
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
                    last_label = None

    except requests.exceptions.RequestException as e:
        print(f"[REQUEST ERROR] {e}")
        label_text = "네트워크오류"
        confidence = 0.0

    # 문장 초기화 타이머
    if output_sentence and (time.time() - last_add_time > RESET_INTERVAL):
        output_sentence = []

    # ====== 시각화 ======
    display_frame = draw_text_korean(
        display_frame,
        f'{label_text} ({confidence:.2f})',
        (10, 30),
        font_size=32,
        color=(0,255,0) if label_text not in ["없음", "무효", "에러", "네트워크오류"] else (0,0,255)
    )
    display_frame = draw_text_korean(display_frame, ' '.join(output_sentence), (10, 70), 28, (255,255,255))

    # 랜드마크 시각화(디버깅용)
    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow("실시간 수어 예측(REST)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()