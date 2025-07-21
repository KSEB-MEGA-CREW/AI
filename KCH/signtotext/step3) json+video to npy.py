import cv2
import mediapipe as mp
import numpy as np
import json
import os
import re

def sanitize_filename(name):
    # 파일 이름으로 사용할 수 없는 문자를 '_'로 대체
    return re.sub(r'[:\/\\?*<>|"]', '_', name)

# 경로 설정
root_folder = r"D:\NIKL_Sign Language Parallel Corpus_2024_LI_SH\2024_0939270-0970090_3053_LI_SH1"
output_dir = r"C:\KEB_bootcamp\project\AI\KCH\signtotext\output_npy\일상생활_학교"
os.makedirs(output_dir, exist_ok=True)

# 추출할 포즈 인덱스 및 기대하는 keypoint 수
POSE_INDEXES = list(range(17))
expected_len = 21*3 + 21*3 + 17*4

# 처리 범위 설정
start_num = 240939270   # 시작 번호 (예시)
stop_num = 240970090    # 종료 번호 (예시)
step = +10

mp_holistic = mp.solutions.holistic

# 랜드마크 추출 함수
def extract_landmarks(landmarks, dims, idxs=None):
    result = []
    if landmarks:
        iter_landmarks = (
            enumerate(landmarks.landmark) if idxs is None
            else ((i, landmarks.landmark[i]) for i in idxs)
        )
        for i, lm in iter_landmarks:
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, 'visibility', 0.0))
            result.extend(coords)
    return result

# 메인 루프
for num in range(start_num, stop_num-1, step):
    base_id = f"VXPAKOKS{num}"
    json_path = os.path.join(root_folder, f"{base_id}.json")
    video_path = os.path.join(root_folder, f"{base_id}.mp4")

    if not (os.path.exists(json_path) and os.path.exists(video_path)):
        print(f"  ⚠️ {base_id} 파일 없음: {json_path} 또는 {video_path}")
        continue

    # JSON 읽기
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = data.get("potogrf", {}).get("fps", 30)
    sign_gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])

    # mp4 전체 프레임 처리
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []
    with mp_holistic.Holistic(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            lh = extract_landmarks(results.left_hand_landmarks, 3)
            rh = extract_landmarks(results.right_hand_landmarks, 3)
            pose = extract_landmarks(results.pose_landmarks, 4, idxs=POSE_INDEXES)
            keypoints = lh + rh + pose
            if len(keypoints) < expected_len:
                keypoints += [0.0] * (expected_len - len(keypoints))
            elif len(keypoints) > expected_len:
                keypoints = keypoints[:expected_len]
            all_keypoints.append(keypoints)
    cap.release()
    all_keypoints = np.array(all_keypoints)

    # gloss 단위로 나눠 저장
    for gloss in sign_gestures:
        gloss_id = str(gloss['gloss_id']).strip()  # ✅ 앞뒤 공백 제거
        gloss_id_clean = sanitize_filename(gloss_id.replace('.npy', '').replace('.NPY', ''))

        start_sec = gloss['start']
        end_sec = gloss['end']
        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        gloss_keypoints = all_keypoints[start_frame:end_frame+1]

        # 라벨별 폴더 저장
        label_dir = os.path.join(output_dir, gloss_id_clean)  # gloss_id_clean은 이미 strip됨
        os.makedirs(label_dir, exist_ok=True)
        out_name = f"{base_id}_{gloss_id_clean}.npy"
        out_path = os.path.join(label_dir, out_name)

        if os.path.exists(out_path):
            print(f"  ⏩ 이미 존재, 건너뜀: {out_path}")
            continue

        np.save(out_path, gloss_keypoints)
        print(f"  ⬇️ 저장: {out_path} (shape={gloss_keypoints.shape})")

print("✅ 변환 완료")