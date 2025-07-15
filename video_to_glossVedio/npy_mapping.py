import os
import cv2
import json
import mediapipe as mp

# strong, weak 폴더 경로
base_dir = os.path.dirname(os.path.abspath(__file__))
target_dirs = ["strong", "weak"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

for target_dir in target_dirs:
    target_path = os.path.join(base_dir, target_dir)
    # strong/weak 폴더 내 모든 하위 폴더(문장 단위)
    for case_folder in os.listdir(target_path):
        case_path = os.path.join(target_path, case_folder)
        if not os.path.isdir(case_path): continue
        # 하위 폴더 내 모든 mp4 파일
        for mp4_file in os.listdir(case_path):
            if not mp4_file.endswith(".mp4"): continue
            mp4_path = os.path.join(case_path, mp4_file)
            print(f"Processing {mp4_path}")
            cap = cv2.VideoCapture(mp4_path)

            frame_kpts = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    keypoints = [
                        {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility
                        }
                        for lm in results.pose_landmarks.landmark
                    ]
                else:
                    keypoints = []
                frame_kpts.append(keypoints)
            cap.release()
            # 결과 저장 (json)
            out_json = mp4_path.replace(".mp4", "_pose.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(frame_kpts, f, ensure_ascii=False, indent=2)
            print(f"Saved pose keypoints: {out_json}")

