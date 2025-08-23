# capture_none_scenarios.py  (지연 핫키 포함 버전)
import os
import time
import cv2
import csv
import numpy as np
from datetime import datetime
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image  # (한글 오버레이용)

# =========================
# 사용자 설정
# =========================
OUTPUT_ROOT = r"C:\want_npy\none_scenarios"   # 시나리오별 폴더가 생성됩니다
CAM_INDEX = 0
WINDOW_FRAMES = 12                  # 저장 프레임 수(학습에서 10프레임 리샘플)
AVG_ZERO_RATIO_MAX = 0.80           # 평균 zero(0.0) 비율 임계치 초과 시 폐기
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"     # 한글 폰트 경로(환경에 맞게 수정 가능)

# 🔸 지연 설정(핫키로 실시간 변경)
PRE_CAPTURE_DELAY_SEC = 1.0
DELAY_MIN, DELAY_MAX = 0.0, 5.0
DELAY_STEP, DELAY_FINE = 0.5, 0.1

# 시나리오 설정: 숫자키 → (폴더명, target개수, allow_hands)
SCENARIOS = {
    "1": {"name": "empty_frame",       "target": 60,  "allow_hands": False},
    "2": {"name": "hands_down",        "target": 80,  "allow_hands": True},
    "3": {"name": "typing_mouse",      "target": 80,  "allow_hands": True},
    "4": {"name": "phone_usage",       "target": 80,  "allow_hands": True},
    "5": {"name": "head_touch_glasses","target": 60,  "allow_hands": True},
    "6": {"name": "look_around",       "target": 60,  "allow_hands": True},
    # 필요 시 7~9도 추가
}

# =========================
# 상수(우리 파이프라인과 동일)
# =========================
POSE_SKIP_INDEXES = set(range(17, 33))   # 하체 제외
EXPECTED_LEN = 21*3 + 21*3 + 17*4        # LH(63)+RH(63)+POSE(17*4)=194

# =========================
# 준비
# =========================
os.makedirs(OUTPUT_ROOT, exist_ok=True)
for cfg in SCENARIOS.values():
    os.makedirs(os.path.join(OUTPUT_ROOT, cfg["name"]), exist_ok=True)

MANIFEST = os.path.join(OUTPUT_ROOT, "manifest.csv")
if not os.path.exists(MANIFEST):
    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","scenario","file","zero_ratio","hands_ratio","accepted"])

KOREAN_FONT = ImageFont.truetype(FONT_PATH, 28)

def draw_text_korean(frame, text, position, font_size=28, color=(255,255,255)):
    font = KOREAN_FONT if font_size == 28 else ImageFont.truetype(FONT_PATH, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(landmarks, dims=3, skip=None):
    out = []
    if landmarks:
        for i, lm in enumerate(landmarks.landmark):
            if skip and i in skip:
                continue
            v = [lm.x, lm.y, lm.z]
            if dims == 4:
                v.append(getattr(lm, 'visibility', 0.0))
            out.extend(v)
    return out

def pad_or_trim(vec, expected=EXPECTED_LEN):
    if len(vec) < expected:
        return vec + [0.0] * (expected - len(vec))
    elif len(vec) > expected:
        return vec[:expected]
    return vec

def save_seq(scenario_name, seq, zero_ratio, hands_ratio):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_name = f"{scenario_name}_{ts}.npy"
    out_path = os.path.join(OUTPUT_ROOT, scenario_name, out_name)
    np.save(out_path, np.asarray(seq, dtype=np.float32))
    with open(MANIFEST, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([ts, scenario_name, out_name, f"{zero_ratio:.4f}", f"{hands_ratio:.4f}", True])
    print(f"  ✅ 저장: {out_path}  (zero={zero_ratio:.2f}, hands={hands_ratio:.2f})")

def log_reject(scenario_name, zero_ratio, hands_ratio, reason):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    with open(MANIFEST, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([ts, scenario_name, "", f"{zero_ratio:.4f}", f"{hands_ratio:.4f}", f"REJECT:{reason}"])
    print(f"  ⚠️ 폐기: {reason}  (zero={zero_ratio:.2f}, hands={hands_ratio:.2f})")

def count_existing(scenario_name):
    folder = os.path.join(OUTPUT_ROOT, scenario_name)
    return len([fn for fn in os.listdir(folder) if fn.endswith(".npy")])

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    global PRE_CAPTURE_DELAY_SEC

    current_key = "2"  # 기본 시나리오(양손 내림)
    saved_counts = {cfg["name"]: count_existing(cfg["name"]) for cfg in SCENARIOS.values()}

    cap = cv2.VideoCapture(CAM_INDEX)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("🎥 NONE(시나리오별) 캡처 시작")
    print(" - 숫자키(1~9): 시나리오 전환")
    print(" - s: 지연 후 12프레임 캡처 & 저장  |  delay=±0.5s(+, -), ±0.1s(], [)")
    print(" - q: 종료\n")
    for k, cfg in SCENARIOS.items():
        print(f"  [{k}] {cfg['name']}  target={cfg['target']}  allow_hands={cfg['allow_hands']}  (현재 {saved_counts[cfg['name']]}개)")

    with mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1
    ) as holistic:
        recording = False
        pending_start = False
        start_due_time = 0.0
        start_requested_at = 0.0

        buffer_frames = []
        hands_count_in_window = 0
        last_switch_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)

            # ===== 현재 시나리오 정보 =====
            cfg = SCENARIOS[current_key]
            scenario = cfg["name"]
            target = cfg["target"]
            allow_hands = cfg["allow_hands"]
            curr_saved = saved_counts[scenario]
            remaining = max(0, target - curr_saved)

            # ===== 랜드마크 추출 → (194,)
            lh = extract_landmarks(res.left_hand_landmarks, dims=3)
            rh = extract_landmarks(res.right_hand_landmarks, dims=3)
            pose = extract_landmarks(res.pose_landmarks, dims=4, skip=POSE_SKIP_INDEXES)
            vec = pad_or_trim(lh + rh + pose)

            # ===== 오버레이 =====
            top_text = f"[{scenario}] {curr_saved}/{target}  (남은 수: {remaining})"
            frame = draw_text_korean(frame, top_text, (10, 10), 28, (0,255,0))
            hint = f"숫자키: 시나리오 전환 | s: 지연 후 캡처 | delay={PRE_CAPTURE_DELAY_SEC:.2f}s (+/-/[/]) | q: 종료"
            frame = draw_text_korean(frame, hint, (10, 44), 24, (255,255,255))

            # 진행 상태 표시
            if pending_start and not recording:
                rem = max(0.0, start_due_time - time.time())
                frame = draw_text_korean(frame, f"⏳ 캡처 시작까지 {rem:0.2f}s", (10, 78), 24, (0,200,255))
            elif recording:
                frame = draw_text_korean(frame, f"● 캡처 중... {len(buffer_frames)}/{WINDOW_FRAMES}", (10, 78), 24, (0,200,0))

            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # ===== 상태/키 입력 =====
            cv2.imshow("Capture NONE (Scenarios)", frame)
            key = cv2.waitKey(1) & 0xFF

            # 시나리오 전환
            if chr(key) in SCENARIOS:
                current_key = chr(key)
                # 전환 시 진행중인 대기/캡처 취소
                recording = False
                pending_start = False
                buffer_frames = []
                hands_count_in_window = 0
                print(f"\n▶ 시나리오 전환: [{current_key}] {SCENARIOS[current_key]['name']}")
                continue

            # 종료
            if key == ord('q'):
                break

            # 🔸 지연 핫키 처리
            delay_changed = False
            if key in (ord('+'), ord('=')):  # 일부 키보드는 Shift+'='이 '+'
                PRE_CAPTURE_DELAY_SEC = clamp(PRE_CAPTURE_DELAY_SEC + DELAY_STEP, DELAY_MIN, DELAY_MAX)
                delay_changed = True
            elif key == ord('-'):
                PRE_CAPTURE_DELAY_SEC = clamp(PRE_CAPTURE_DELAY_SEC - DELAY_STEP, DELAY_MIN, DELAY_MAX)
                delay_changed = True
            elif key == ord(']'):
                PRE_CAPTURE_DELAY_SEC = clamp(PRE_CAPTURE_DELAY_SEC + DELAY_FINE, DELAY_MIN, DELAY_MAX)
                delay_changed = True
            elif key == ord('['):
                PRE_CAPTURE_DELAY_SEC = clamp(PRE_CAPTURE_DELAY_SEC - DELAY_FINE, DELAY_MIN, DELAY_MAX)
                delay_changed = True

            if delay_changed:
                print(f"⏱ 지연 = {PRE_CAPTURE_DELAY_SEC:.2f}s")
                # 카운트다운 중이면 새 지연을 즉시 반영
                if pending_start:
                    start_due_time = start_requested_at + PRE_CAPTURE_DELAY_SEC
                    # 지연이 0이 되었으면 즉시 시작
                    if PRE_CAPTURE_DELAY_SEC <= 0.0:
                        start_due_time = time.time()

            # target 달성 시 안내만 하고 skip
            if curr_saved >= target:
                if time.time() - last_switch_time > 2.0:
                    print(f"✔ [{scenario}] 목표 달성({curr_saved}/{target}). 다른 시나리오로 전환하세요.")
                    last_switch_time = time.time()
                # 진행 중이던 대기도 취소
                pending_start = False
                recording = False
                continue

            # s 눌러 "지연 캡처" 예약
            if key == ord('s') and (not recording) and (not pending_start):
                pending_start = True
                start_requested_at = time.time()
                start_due_time = start_requested_at + PRE_CAPTURE_DELAY_SEC
                # 지연이 0이면 즉시 시작되도록
                if PRE_CAPTURE_DELAY_SEC <= 0.0:
                    start_due_time = time.time()
                buffer_frames = []
                hands_count_in_window = 0
                print(f"▶ {PRE_CAPTURE_DELAY_SEC:.2f}s 후 캡처 시작 예정 — 준비하세요.")

            # 대기 시간이 끝나면 녹화 시작
            if pending_start and (time.time() >= start_due_time):
                pending_start = False
                recording = True
                buffer_frames = []
                hands_count_in_window = 0
                print("▶ 캡처 시작(12프레임)")

            # 캡처 중이면 프레임 누적
            if recording:
                buffer_frames.append(vec)
                if (len(lh) > 0) or (len(rh) > 0):
                    hands_count_in_window += 1

                if len(buffer_frames) >= WINDOW_FRAMES:
                    arr = np.asarray(buffer_frames, dtype=np.float32)  # (12, 194)
                    zero_ratio = float(np.count_nonzero(arr == 0.0)) / float(arr.size)
                    hands_ratio = hands_count_in_window / float(WINDOW_FRAMES)

                    # 품질 1: zero 비율
                    if zero_ratio > AVG_ZERO_RATIO_MAX:
                        log_reject(scenario, zero_ratio, hands_ratio, "too_many_zeros")
                    # 품질 2: allow_hands 규칙 위반
                    elif (not allow_hands) and (hands_ratio > 0.05):
                        log_reject(scenario, zero_ratio, hands_ratio, "hands_detected_in_nohands_scenario")
                    else:
                        save_seq(scenario, arr, zero_ratio, hands_ratio)
                        saved_counts[scenario] += 1

                    # 윈도 종료
                    recording = False
                    buffer_frames = []
                    hands_count_in_window = 0
                    print(f"⏹ 캡처 종료. 남은 수: {max(0, target - saved_counts[scenario])}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 완료")
    for k, cfg in SCENARIOS.items():
        name = cfg["name"]
        print(f" - {name}: {saved_counts[name]}/{cfg['target']}개")
    print(f"📄 manifest: {MANIFEST}")

if __name__ == "__main__":
    main()