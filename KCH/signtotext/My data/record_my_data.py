import cv2
import mediapipe as mp
import numpy as np
import time
import os
from collections import defaultdict
from PIL import ImageFont, ImageDraw, Image

# ===== 사용자 수정 구간 =====
SAVE_ROOT = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\My data\dataset\찬혁"  # 최상위 폴더
LABEL_NAME = "이름1"    # ⭐️ 라벨명 지정
LABEL_DIR = os.path.join(SAVE_ROOT, LABEL_NAME)
os.makedirs(LABEL_DIR, exist_ok=True)

POSE_SKIP_INDEXES = set(range(17, 33))
EXPECTED_LEN = 194
SAVE_FRAMES = 20
STABLE_SKIP_FRAMES = 4
WAIT_SEC = 1.0
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

# ===== 라벨별 개수 집계 함수 =====
def count_label_files(label_dir):
    cnt = len([fname for fname in os.listdir(label_dir) if fname.endswith('.npy')])
    return cnt

def print_label_count(label, count):
    print(f"\n[📊 '{label}' 저장 개수: {count}개]")

# ===== 한글 텍스트 출력 함수 (PIL) =====
def draw_text_korean(frame, text, position, font_size=30, color=(255,255,255)):
    font = ImageFont.truetype(FONT_PATH, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ===== MediaPipe 초기화 =====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
print(f"▶ 's' 키로 수집 시작, 'q' 키로 종료 ({LABEL_NAME} 폴더로 저장)")

label_count = count_label_files(LABEL_DIR)
print_label_count(LABEL_NAME, label_count)

state = "idle"
start_time = None
data_buffer = []

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    display_frame = frame.copy()

    # 랜드마크 시각화
    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    current_time = time.time()

    if state == "waiting":
        elapsed = current_time - start_time
        display_frame = draw_text_korean(display_frame, f"⏱️ 대기 중: {elapsed:.1f}초", (30, 30), font_size=32, color=(0,0,255))
        if elapsed >= WAIT_SEC:
            state = "collecting"
            print("📸 수집 시작!")

    elif state == "collecting":
        lh = extract_landmarks(results.left_hand_landmarks)
        rh = extract_landmarks(results.right_hand_landmarks)
        pose = extract_landmarks(results.pose_landmarks, dims=4, skip=POSE_SKIP_INDEXES)

        keypoints = lh + rh + pose
        if len(keypoints) < EXPECTED_LEN:
            keypoints += [0.0] * (EXPECTED_LEN - len(keypoints))
        elif len(keypoints) > EXPECTED_LEN:
            keypoints = keypoints[:EXPECTED_LEN]

        data_buffer.append(keypoints)
        display_frame = draw_text_korean(
            display_frame,
            f"수집 중: {len(data_buffer)}/{SAVE_FRAMES + STABLE_SKIP_FRAMES}",
            (30, 30), font_size=32, color=(0,255,0))

        if len(data_buffer) >= SAVE_FRAMES + STABLE_SKIP_FRAMES:
            output_array = np.array(data_buffer[STABLE_SKIP_FRAMES:])
            filename = f"{int(time.time())}_{LABEL_NAME}.npy"
            save_path = os.path.join(LABEL_DIR, filename)
            np.save(save_path, output_array)
            print(f"✅ 저장 완료: {save_path} (shape={output_array.shape})")
            label_count = count_label_files(LABEL_DIR)
            print_label_count(LABEL_NAME, label_count)
            data_buffer = []
            state = "idle"
            start_time = None

    elif state == "idle":
        display_frame = draw_text_korean(display_frame, "▶ 's' 누르면 수집 시작", (30, 30), font_size=32, color=(255,255,255))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and state == "idle":
        state = "waiting"
        start_time = time.time()
        data_buffer = []
        print("🟢 1초 대기 후 수집을 시작합니다...")
    elif key == ord('q'):
        print("❎ 종료")
        break

    cv2.imshow(f"실시간 수어 데이터 수집 ({LABEL_NAME})", display_frame)

cap.release()
cv2.destroyAllWindows()