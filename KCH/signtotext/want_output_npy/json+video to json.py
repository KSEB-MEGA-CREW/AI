import cv2
import mediapipe as mp
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== 경로 설정 ======
root_folder = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\dataset\test"   # 변환할 폴더(최상위)
output_dir = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\dataset\video to json"
os.makedirs(output_dir, exist_ok=True)

POSE_INDEXES = list(range(17))
BUFFER_FRAMES = 5

hand_landmarks = [
    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]
pose_landmarks = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST"
]

def sanitize_filename(name):
    return re.sub(r'[:\/\\?*<>|"]', '_', name)

def extract_hand_json(hand_obj):
    if hand_obj is None:
        return [{"name": name, "x": 0.0, "y": 0.0, "z": 0.0} for name in hand_landmarks]
    return [
        {"name": hand_landmarks[i], "x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
        for i, lm in enumerate(hand_obj.landmark)
    ]

def extract_pose_json(pose_obj):
    if pose_obj is None:
        return [{"name": name, "x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0} for name in pose_landmarks]
    out = []
    for i, lm in enumerate(pose_obj.landmark):
        if i in POSE_INDEXES and i < 17:
            out.append({
                "name": pose_landmarks[i],
                "x": float(lm.x), "y": float(lm.y), "z": float(lm.z),
                "visibility": float(getattr(lm, 'visibility', 0.0))
            })
    return out

def process_one_gloss_json(task):
    base_id = task["base_id"]
    video_path = task["video_path"]
    gloss = task["gloss"]
    fps = task["fps"]
    gloss_id = str(gloss['gloss_id']).strip()
    gloss_id_clean = sanitize_filename(gloss_id)
    label_dir = os.path.join(output_dir, gloss_id_clean)
    os.makedirs(label_dir, exist_ok=True)
    out_name = f"{base_id}_{gloss_id_clean}.json"
    out_path = os.path.join(label_dir, out_name)

    # 이미 존재하면 스킵
    if os.path.exists(out_path):
        return (gloss_id, False, out_path)

    start_sec = gloss['start']
    end_sec = gloss['end']
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    buffer_start = max(start_frame - BUFFER_FRAMES, 0)

    results_json = []
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as holistic:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, buffer_start)
        for frame_idx in range(buffer_start, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            if frame_idx < start_frame:
                continue  # buffer 프레임은 저장 X

            frame_dict = {
                "left_hand": extract_hand_json(results.left_hand_landmarks),
                "right_hand": extract_hand_json(results.right_hand_landmarks),
                "pose": extract_pose_json(results.pose_landmarks)
            }
            results_json.append(frame_dict)
        cap.release()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    return (gloss_id, True, out_path)

# ===== json/mp4 쌍 자동 수집 =====
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

# ===== 태스크 목록 만들기 (라벨별로 변환 제한 등 옵션 추가 가능) =====
task_list = []
for base_id, json_path, video_path in all_json_paths:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = data.get("potogrf", {}).get("fps", 30)
    sign_gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])
    for gloss in sign_gestures:
        gloss_id = str(gloss['gloss_id']).strip()
        task_list.append({
            "base_id": base_id,
            "json_path": json_path,
            "video_path": video_path,
            "gloss": gloss,
            "fps": fps
        })

# ===== 병렬 변환 실행 =====
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

final_counts = Counter()
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_one_gloss_json, task): task for task in task_list}
    for future in as_completed(futures):
        gloss_id, saved, out_path = future.result()
        if saved:
            final_counts[gloss_id] += 1
            print(f"  ⬇️ 저장: {out_path}")
        else:
            print(f"  ⏩ 이미 존재, 건너뜀: {out_path}")

print("\n🎉 전체 변환 완료 (저장 기준)")
print("📊 새로 저장된 파일 개수:")
for gloss_id, cnt in final_counts.items():
    print(f"  - {gloss_id}: {cnt}개 저장됨")