import cv2
import mediapipe as mp
import numpy as np
import json
import os
import re

# 🔧 사용자 설정
TARGET_LABELS = ["지시2","요리1","잘하다2","무엇1","지시1#","좋다1","지시2"]
PER_LABEL_LIMIT = 829
converted_counts = {label: 0 for label in TARGET_LABELS}
BUFFER_FRAMES = 5  # MediaPipe 안정화를 위한 버퍼 프레임 수

# 📂 경로 설정
root_folder = r"D:"
output_dir = r"C:\want_output_npy"
os.makedirs(output_dir, exist_ok=True)

POSE_INDEXES = list(range(17))
expected_len = 21*3 + 21*3 + 17*4

mp_holistic = mp.solutions.holistic

def sanitize_filename(name):
    return re.sub(r'[:\/\\?*<>|"]', '_', name)

def extract_landmarks(landmarks, dims, idxs=None):
    result = []
    if landmarks:
        iter_landmarks = (
            enumerate(landmarks.landmark) if idxs is None
            else ((i, landmarks.landmark[i]) for i in idxs)
        )
        for _, lm in iter_landmarks:
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, 'visibility', 0.0))
            result.extend(coords)
    return result

# 🔍 전체 폴더 내 json/mp4 탐색 (하위 폴더 포함)
all_json_paths = []
for dirpath, _, filenames in os.walk(root_folder):
    for fname in filenames:
        if fname.endswith(".json"):
            base = os.path.splitext(fname)[0]
            json_path = os.path.join(dirpath, f"{base}.json")
            video_path = os.path.join(dirpath, f"{base}.mp4")
            if os.path.exists(video_path):
                all_json_paths.append((base, json_path, video_path))

print(f"🔍 총 {len(all_json_paths)}개 JSON/MP4 세트 발견됨")

# 🔁 처리 시작
for base_id, json_path, video_path in all_json_paths:

    if all(converted_counts[label] >= PER_LABEL_LIMIT for label in TARGET_LABELS):
        print("\n✅ 모든 라벨 변환 완료 (새로 저장 기준)")
        break

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = data.get("potogrf", {}).get("fps", 30)
    sign_gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])

    for gloss in sign_gestures:
        gloss_id = str(gloss['gloss_id']).strip()
        if gloss_id not in TARGET_LABELS:
            continue
        if converted_counts[gloss_id] >= PER_LABEL_LIMIT:
            continue

        gloss_id_clean = sanitize_filename(gloss_id)
        label_dir = os.path.join(output_dir, gloss_id_clean)
        os.makedirs(label_dir, exist_ok=True)
        out_name = f"{base_id}_{gloss_id_clean}.npy"
        out_path = os.path.join(label_dir, out_name)

        if os.path.exists(out_path):
            print(f"  ⏩ 이미 존재, 건너뜀: {out_path}")
            continue

        start_sec = gloss['start']
        end_sec = gloss['end']
        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        buffer_start = max(start_frame - BUFFER_FRAMES, 0)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, buffer_start)

        gloss_keypoints = []
        with mp_holistic.Holistic(
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        ) as holistic:
            for frame_idx in range(buffer_start, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                if frame_idx < start_frame:
                    continue  # 🎯 buffer 프레임은 저장하지 않음

                lh = extract_landmarks(results.left_hand_landmarks, 3)
                rh = extract_landmarks(results.right_hand_landmarks, 3)
                pose = extract_landmarks(results.pose_landmarks, 4, idxs=POSE_INDEXES)
                keypoints = lh + rh + pose
                if len(keypoints) < expected_len:
                    keypoints += [0.0] * (expected_len - len(keypoints))
                elif len(keypoints) > expected_len:
                    keypoints = keypoints[:expected_len]
                gloss_keypoints.append(keypoints)
        cap.release()

        gloss_keypoints = np.array(gloss_keypoints)
        np.save(out_path, gloss_keypoints)
        converted_counts[gloss_id] += 1
        print(f"  ⬇️ 저장: {out_path} (shape={gloss_keypoints.shape})")

# ✅ 최종 결과 출력
print("\n🎉 전체 변환 완료 (저장 기준)")
print("📊 새로 저장된 파일 개수:")
for label, count in converted_counts.items():
    print(f"  - {label}: {count}개 저장됨")