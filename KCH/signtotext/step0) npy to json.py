import numpy as np
import json

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
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY"
]

npy_path = "your_data.npy"
data = np.load(npy_path)  # (frame, 194)

results = []
for frame in data:
    frame_dict = {}
    left_hand_arr = frame[0:63].reshape(21, 3)
    left_hand = [
        {"name": hand_landmarks[i], "x": float(x), "y": float(y), "z": float(z)}
        for i, (x, y, z) in enumerate(left_hand_arr)
    ]
    right_hand_arr = frame[63:126].reshape(21, 3)
    right_hand = [
        {"name": hand_landmarks[i], "x": float(x), "y": float(y), "z": float(z)}
        for i, (x, y, z) in enumerate(right_hand_arr)
    ]
    pose_arr = frame[126:194].reshape(17, 4)
    pose = [
        {"name": pose_landmarks[i], "x": float(x), "y": float(y), "z": float(z), "visibility": float(v)}
        for i, (x, y, z, v) in enumerate(pose_arr)
    ]
    frame_dict["left_hand"] = left_hand
    frame_dict["right_hand"] = right_hand
    frame_dict["pose"] = pose
    results.append(frame_dict)

with open("converted.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ 변환 완료! (예시 파일: converted.json)")