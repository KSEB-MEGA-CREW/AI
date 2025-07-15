import cv2
import mediapipe as mp
import numpy as np
import json
import os
import re

# 🔸 파일명 안전하게 만들기
def sanitize_filename(name):
    return re.sub(r'[:\/\\?*<>|"]', '_', name)

# 🔸 설정
folder = "2024_LI_DC_0779230-0789690_1035"  # 원본 폴더 경로
output_dir = "등수"                          # npy 저장 경로
os.makedirs(output_dir, exist_ok=True)

POSE_INDEXES = list(range(17))              # ✅ 포즈: 0~16번만 사용
expected_len = 21*3 + 21*3 + 17*4            # Left Hand + Right Hand + Pose(0~16) = 194

# 🔸 파일ID 설정
start_num = 240779260                        # VXPAKOKS 뒤 번호
file_count = 1                               # 처리할 파일 수 (1개 처리 시 = 1)

# 🔸 keypoint 추출 함수
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

# 🔸 본격 처리 루프
for i in range(file_count):
    base_id = f"VXPAKOKS{start_num + 10*i}"
    for cam, cam_label in zip(['', 'L', 'R'], ['C', 'L', 'R']):
        json_file = os.path.join(folder, f"{base_id}.json")
        video_file = os.path.join(folder, f"{base_id}{cam}.mp4")
        if not (os.path.exists(json_file) and os.path.exists(video_file)):
            print(f"⚠️ 파일 없음: {json_file} 또는 {video_file}")
            continue

        print(f"▶ 처리중: {base_id}{cam} ({cam_label})")

        # 🔸 JSON 파싱
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        fps = data["potogrf"]["fps"]
        sign_gestures = data["sign_script"]["sign_gestures_strong"]

        # 🔸 비디오 프레임 단위로 keypoints 추출
        cap = cv2.VideoCapture(video_file)
        all_keypoints = []
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(min_detection_confidence=0.7,
                                   min_tracking_confidence=0.7) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                lh = extract_landmarks(results.left_hand_landmarks, 3)
                rh = extract_landmarks(results.right_hand_landmarks, 3)
                pose = extract_landmarks(results.pose_landmarks, 4, idxs=POSE_INDEXES)  # ✅ 0~16번 포즈만

                keypoints = lh + rh + pose
                if len(keypoints) < expected_len:
                    keypoints += [0.0] * (expected_len - len(keypoints))
                elif len(keypoints) > expected_len:
                    keypoints = keypoints[:expected_len]
                all_keypoints.append(keypoints)

        cap.release()
        all_keypoints = np.array(all_keypoints)  # (frames, 194)

        # 🔸 gloss별 잘라서 npy 저장
        for gloss in sign_gestures:
            start_sec = gloss['start']
            end_sec = gloss['end']
            gloss_id = gloss['gloss_id']
            gloss_id_clean = sanitize_filename(str(gloss_id).replace('.npy', '').replace('.NPY', ''))

            start_frame = int(round(start_sec * fps))
            end_frame = int(round(end_sec * fps))
            gloss_keypoints = all_keypoints[start_frame:end_frame+1]

            out_name = f"{base_id}_{gloss_id_clean}_{cam_label}.npy"
            out_path = os.path.join(output_dir, out_name)
            np.save(out_path, gloss_keypoints)
            print(f"  ⬇️ 저장: {out_path} (shape={gloss_keypoints.shape})")