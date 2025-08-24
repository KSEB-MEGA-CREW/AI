import cv2
import mediapipe as mp
import numpy as np
import json
import os
import re
import math
from pathlib import Path

# ==== 사용자 설정 ====
with open("/home/202044005/KSEB/text_to_word/preprocessed_data/unique_glosses.json", 'r', encoding='utf-8') as f:
    TARGET_LABELS = json.load(f)



# TARGET_LABELS = ["가다1","오다1","집1","물건2","결정1","준비1","원인1","이전1","때2","먹다1"]
PER_LABEL_LIMIT = 500
BUFFER_FRAMES = 5
MIN_VALID_FRAMES = 7          # 🔸 너무 짧은 구간은 버림
APPLY_EMA = False             # 🔸 튐 프레임 완화(지수이동평균)
EMA_ALPHA = 0.5
APPLY_SPIKE_GUARD = True      # 🔸 2~3프레임 연속 큰 점프면 세그먼트 폐기
SPIKE_THRESH = 0.35           #   (0~1 화면좌표 기준 대략값, 필요시 조정)
SPIKE_RUN = 2
NORMALIZE_TO_SHOULDERS = False # 🔸 어깨 중심 정규화(기존 학습과 다르면 False)

# ==== 경로 설정 ====
root_folder = r"/storage/202044005/data"
output_dir = r"/storage/202044005/numpy"
os.makedirs(output_dir, exist_ok=True)

POSE_INDEXES = list(range(17))  # 0..16 (얼굴+어깨/팔 위주, 17~는 제외)
expected_len = 21*3 + 21*3 + 17*4  # = 194

mp_holistic = mp.solutions.holistic

def sanitize_filename(name):
    return re.sub(r'[:\/\\?*<>|"]', '_', name)

def extract_landmarks(landmarks, dims, idxs=None):
    result = []
    if landmarks:
        it = (enumerate(landmarks.landmark) if idxs is None
            else ((i, landmarks.landmark[i]) for i in idxs))
        for _, lm in it:
            coords = [lm.x, lm.y, lm.z]
            if dims == 4:
                coords.append(getattr(lm, 'visibility', 0.0))
            result.extend(coords)
    return result

def ema_smooth(seq, alpha=EMA_ALPHA):
    """seq: (T, D) numpy. 간단 EMA. 0만인 프레임은 그대로 둠."""
    if len(seq) == 0:
        return seq
    out = seq.copy()
    for t in range(1, len(seq)):
        prev = out[t-1]
        curr = seq[t]
        # 모두 0인 프레임은 스킵(검출 실패 프레임)
        if not np.any(curr):
            out[t] = curr
        else:
            out[t] = alpha * curr + (1 - alpha) * prev
    return out

def has_spike(seq, thresh=SPIKE_THRESH, run=SPIKE_RUN):
    """연속 run 프레임 이상 큰 점프가 있는지 검사(손+포즈 전체 L2)."""
    if len(seq) < run + 1:
        return False
    # 좌표만(visibility 제외) 사용: 21*3 + 21*3 + 17*3 = 63+63+51 = 177
    # 아래는 visibility 포함 194에서 마지막 17개를 제외
    coord_dim = expected_len - 17  # 177
    diffs = np.linalg.norm(np.diff(seq[:, :coord_dim], axis=0), axis=1)
    # 0만인 프레임들 사이 diff는 작게 나올 수 있으므로, 비제로 프레임들 중심으로 판단하면 더 정교함
    count = 0
    for d in diffs:
        if d > thresh:
            count += 1
            if count >= run:
                return True
        else:
            count = 0
    return False

def normalize_by_shoulders(frame_vec):
    """
    frame_vec: (194,) [lh(63) + rh(63) + pose(68)]
    포즈에서 좌/우 어깨(landmark 11,12)의 x,y를 사용해 중심/스케일 정규화.
    z/visibility는 그대로 둠.
    """
    if len(frame_vec) != expected_len:
        return frame_vec
    lh = frame_vec[0:63]
    rh = frame_vec[63:126]
    pose = frame_vec[126:]  # 17*4

    # pose에서 11,12번째(글로벌 인덱스 기준)가 포함돼 있음(0..16 중 11,12 존재)
    # pose 블록 내에서 각 포인트는 [x,y,z,vis] 순으로 4개씩
    def get_xy(pose_block, idx_in_pose):
        base = idx_in_pose * 4
        return pose_block[base], pose_block[base+1]

    try:
        lx, ly = get_xy(pose, 11)  # left shoulder
        rx, ry = get_xy(pose, 12)  # right shoulder
        cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0
        scale = max(np.hypot(rx - lx, ry - ly), 1e-6)

        def norm_xy_block(block, step):
            out = block.copy()
            # step=3 for hands, step=4 for pose (xy는 각 포인트의 처음 2개)
            for i in range(0, len(block), step):
                out[i]   = (block[i]   - cx) / scale
                out[i+1] = (block[i+1] - cy) / scale
            return out

        lh2 = norm_xy_block(lh, 3)
        rh2 = norm_xy_block(rh, 3)
        pose2 = pose.copy()
        for i in range(0, len(pose), 4):
            pose2[i]   = (pose[i]   - cx) / scale
            pose2[i+1] = (pose[i+1] - cy) / scale
        return np.concatenate([lh2, rh2, pose2], axis=0)
    except Exception:
        return frame_vec

# ==== 전체 폴더 내 json/mp4 탐색 ====
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

# 라벨별 변환 개수 카운터
converted_counts = {label: 0 for label in TARGET_LABELS}

# ==== 처리 ====
for base_id, json_path, video_path in all_json_paths:

    # 모든 라벨 상한 채우면 종료
    if all(converted_counts[lbl] >= PER_LABEL_LIMIT for lbl in TARGET_LABELS):
        print("\n✅ 모든 라벨 변환 완료 (저장 기준)")
        break

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = data.get("potogrf", {}).get("fps", 30) or 30
    signs = data.get("sign_script", {}).get("sign_gestures_strong", [])
    if not signs:
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패: {video_path}")
        continue

    with mp_holistic.Holistic(min_detection_confidence=0.7,
                                min_tracking_confidence=0.7,
                                model_complexity=1) as holistic:  # 🔸 성능/속도 균형
        for seg_idx, gloss in enumerate(signs, start=1):
            gloss_id = str(gloss.get('gloss_id', '')).strip()
            if gloss_id not in TARGET_LABELS:
                continue
            if converted_counts[gloss_id] >= PER_LABEL_LIMIT:
                continue

            # 파일명 충돌 방지: start~end 프레임 포함
            start_sec = float(gloss['start'])
            end_sec = float(gloss['end'])
            start_frame = int(math.floor(start_sec * fps))
            end_frame   = int(math.ceil(end_sec * fps))
            buffer_start = max(start_frame - BUFFER_FRAMES, 0)

            gloss_id_clean = sanitize_filename(gloss_id)
            label_dir = os.path.join(output_dir, gloss_id_clean)
            os.makedirs(label_dir, exist_ok=True)

            out_name = f"{base_id}_{gloss_id_clean}_{start_frame:06d}-{end_frame:06d}.npy"
            out_path = os.path.join(label_dir, out_name)
            if os.path.exists(out_path):
                print(f"  ⏩ 이미 존재, 건너뜀: {out_path}")
                continue

            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, buffer_start)
                frames = []
                for frame_idx in range(buffer_start, end_frame + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    if frame_idx < start_frame:
                        continue

                    lh = extract_landmarks(results.left_hand_landmarks, 3)
                    rh = extract_landmarks(results.right_hand_landmarks, 3)
                    pose = extract_landmarks(results.pose_landmarks, 4, idxs=POSE_INDEXES)

                    keypoints = lh + rh + pose
                    if len(keypoints) < expected_len:
                        keypoints += [0.0] * (expected_len - len(keypoints))
                    elif len(keypoints) > expected_len:
                        keypoints = keypoints[:expected_len]

                    frames.append(keypoints)

                seq = np.array(frames, dtype=np.float32)  # 🔸 float32로 저장 용량 절감
                # 최소 길이 필터
                if len(seq) < MIN_VALID_FRAMES:
                    print(f"  ⚠️ 너무 짧음({len(seq)}프레임), 건너뜀: {out_name}")
                    continue

                # 스파이크 감지 → 폐기(옵션)
                if APPLY_SPIKE_GUARD and has_spike(seq, SPIKE_THRESH, SPIKE_RUN):
                    print(f"  ⚠️ 좌표 스파이크 감지, 건너뜀: {out_name}")
                    continue

                # EMA 스무딩(옵션)
                if APPLY_EMA:
                    seq = ema_smooth(seq)

                # 어깨 정규화(옵션)
                if NORMALIZE_TO_SHOULDERS:
                    seq = np.stack([normalize_by_shoulders(f) for f in seq], axis=0)

                np.save(out_path, seq)
                converted_counts[gloss_id] += 1
                print(f"  ⬇️ 저장: {out_path} (shape={seq.shape})")

            except Exception as e:
                print(f"  ❌ 세그먼트 처리 오류({out_name}): {e}")
                continue

    cap.release()

# ==== 요약 ====
print("\n🎉 전체 변환 완료 (저장 기준)")
print("📊 새로 저장된 파일 개수:")
for label, count in converted_counts.items():
    print(f"  - {label}: {count}개 저장됨")