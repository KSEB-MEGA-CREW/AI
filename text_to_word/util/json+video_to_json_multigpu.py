
import cv2
# import mediapipe as mp # This will be imported inside the process function
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

# ====== 경로 설정 ======
root_folder = r"/storage/202044005/NIKL_Sign Language Parallel Corpus_2024_BD_PP"   # 변환할 폴더(최상위)
output_dir = r"/storage/202044005/video_to_json/NIKL_Sign Language Parallel Corpus_2024_BD_PP"
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
    # Sanitize the filename by replacing invalid characters.
    return re.sub(r'[:/\\?*<>|"]', '_', name)

def extract_hand_json(hand_obj):
    # Extract hand landmarks into a JSON-serializable format.
    if hand_obj is None:
        # Return a default structure if no hand is detected.
        return [{"name": name, "x": 0.0, "y": 0.0, "z": 0.0} for name in hand_landmarks]
    # Return the list of landmark dictionaries.
    return [
        {"name": hand_landmarks[i], "x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} 
        for i, lm in enumerate(hand_obj.landmark)
    ]

def extract_pose_json(pose_obj):
    # Extract pose landmarks into a JSON-serializable format.
    if pose_obj is None:
        # Return a default structure if no pose is detected.
        return [{"name": name, "x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0} for name in pose_landmarks]
    out = []
    # Iterate through landmarks and extract required ones.
    for i, lm in enumerate(pose_obj.landmark):
        if i in POSE_INDEXES and i < 17:
            # Append landmark data to the output list.
            out.append({
                "name": pose_landmarks[i],
                "x": float(lm.x), "y": float(lm.y), "z": float(lm.z),
                "visibility": float(getattr(lm, 'visibility', 0.0))
            })
    return out

def process_one_gloss_json(task, gpu_id):
    # Set the visible GPU device for this specific process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Import mediapipe here so it respects the environment variable.
    import mediapipe as mp

    # Unpack the task dictionary.
    base_id = task["base_id"]
    video_path = task["video_path"]
    gloss = task["gloss"]
    fps = task["fps"]
    # Sanitize the gloss_id to use as a directory name.
    gloss_id = str(gloss['gloss_id']).strip()
    gloss_id_clean = sanitize_filename(gloss_id)
    # Create a directory for the current gloss label.
    label_dir = os.path.join(output_dir, gloss_id_clean)
    os.makedirs(label_dir, exist_ok=True)
    # Define the output path for the JSON file.
    out_name = f"{base_id}_{gloss_id_clean}.json"
    out_path = os.path.join(label_dir, out_name)

    # Skip if the file already exists.
    if os.path.exists(out_path):
        return (gloss_id, False, out_path)

    # Calculate start and end frames for the gloss.
    start_sec = gloss['start']
    end_sec = gloss['end']
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    # Define a buffer to capture context around the gloss.
    buffer_start = max(start_frame - BUFFER_FRAMES, 0)

    results_json = []
    # Initialize MediaPipe Holistic model.
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as holistic:
        # Open the video file.
        cap = cv2.VideoCapture(video_path)
        # Set the starting frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, buffer_start)
        # Process each frame from buffer_start to end_frame.
        for frame_idx in range(buffer_start, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert the frame to RGB for MediaPipe.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the image to get holistic landmarks.
            results = holistic.process(image_rgb)

            # Skip saving buffer frames.
            if frame_idx < start_frame:
                continue

            # Create a dictionary for the current frame's landmarks.
            frame_dict = {
                "left_hand": extract_hand_json(results.left_hand_landmarks),
                "right_hand": extract_hand_json(results.right_hand_landmarks),
                "pose": extract_pose_json(results.pose_landmarks)
            }
            # Append the frame data to our results list.
            results_json.append(frame_dict)
        # Release the video capture object.
        cap.release()

    # Write the extracted landmarks to a JSON file.
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    # Return the result of the operation.
    return (gloss_id, True, out_path)

# ===== json/mp4 쌍 자동 수집 =====
all_json_paths = []
# Walk through the root folder to find all json files.
for dirpath, _, filenames in os.walk(root_folder):
    for fname in filenames:
        if fname.endswith(".json"):
            # Get the base name of the file.
            base = os.path.splitext(fname)[0]
            # Construct paths for json and mp4 files.
            json_path = os.path.join(dirpath, f"{base}.json")
            video_path = os.path.join(dirpath, f"{base}.mp4")
            # Add to the list if the corresponding video exists.
            if os.path.exists(video_path):
                all_json_paths.append((base, json_path, video_path))

# Print the total number of file sets found.
print(f"🔍 총 {len(all_json_paths)}개 JSON/MP4 세트 발견됨")

# ===== 태스크 목록 만들기 (라벨별로 변환 제한 등 옵션 추가 가능) =====
task_list = []
# Iterate through all the found json/mp4 pairs.
for base_id, json_path, video_path in all_json_paths:
    # Open and load the json file.
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Get the frames per second (fps) from the json data.
    fps = data.get("potogrf", {}).get("fps", 30)
    # Get the list of sign gestures.
    sign_gestures = data.get("sign_script", {}).get("sign_gestures_strong", [])
    # Create a task for each sign gesture.
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
# Define the number of GPUs to use.
NUM_GPUS = 2
# Create a cycle iterator for GPU ids.
gpu_ids = itertools.cycle(range(NUM_GPUS))
# Initialize a counter for the final results.
final_counts = Counter()
# Use ProcessPoolExecutor for true parallelism.
with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
    # Submit tasks to the executor, assigning a GPU to each.
    futures = {executor.submit(process_one_gloss_json, task, next(gpu_ids)): task for task in task_list}
    # Process futures as they complete.
    for future in as_completed(futures):
        # Get the result from the future.
        gloss_id, saved, out_path = future.result()
        # Check if the file was saved or skipped.
        if saved:
            final_counts[gloss_id] += 1
            print(f"  ⬇️ 저장: {out_path}")
        else:
            print(f"  ⏩ 이미 존재, 건너뜀: {out_path}")

# Print a summary of the conversion process.
print("\n🎉 전체 변환 완료 (저장 기준)")
print("📊 새로 저장된 파일 개수:")
# Print the counts for each gloss_id.
for gloss_id, cnt in final_counts.items():
    print(f"  - {gloss_id}: {cnt}개 저장됨")
