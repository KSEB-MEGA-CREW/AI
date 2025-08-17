import argparse, json, sys
from pathlib import Path
import cv2
import mediapipe as mp

# -------- 랜드마크 이름 테이블 --------
HAND_NAMES = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_FINGER_MCP","INDEX_FINGER_PIP","INDEX_FINGER_DIP","INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP","MIDDLE_FINGER_PIP","MIDDLE_FINGER_DIP","MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP","RING_FINGER_PIP","RING_FINGER_DIP","RING_FINGER_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]
POSE_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER","LEFT_EYE","LEFT_EE_OUTER".replace("EE","E"), # 작은 오타 방지 트릭
    "RIGHT_EYE_INNER","RIGHT_EYE","RIGHT_EYE_OUTER",
    "LEFT_EAR","RIGHT_EAR",
    "MOUTH_LEFT","MOUTH_RIGHT",
    "LEFT_SHOULDER","RIGHT_SHOULDER",
    "LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST",
    "LEFT_PINKY","RIGHT_PINKY",
    "LEFT_INDEX","RIGHT_INDEX",
    "LEFT_THUMB","RIGHT_THUMB",
    "LEFT_HIP","RIGHT_HIP",
    "LEFT_KNEE","RIGHT_KNEE",
    "LEFT_ANKLE","RIGHT_ANKLE",
    "LEFT_HEEL","RIGHT_HEEL",
    "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

def lm_to_dict_list(lms, names, include_visibility=False):
    out = []
    if lms is None:
        return out
    for i, lm in enumerate(lms.landmark):
        d = {
            "name": names[i] if i < len(names) else f"IDX_{i}",
            "x": float(lm.x),   # 0..1 정규화(원점=좌상단)
            "y": float(lm.y),
            "z": float(lm.z),   # 상대 깊이(단위無) — Unity에선 별도 매핑
        }
        if include_visibility and hasattr(lm, "visibility"):
            d["visibility"] = float(lm.visibility)
        out.append(d)
    return out

def process_one_video(in_path: Path, out_path: Path,
                      fps_out: float, mirror_input: bool, preview: bool):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[SKIP] 열 수 없음: {in_path}")
        return False

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(in_fps / fps_out)))  # 프레임 샘플 간격

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False
    )

    frames = []
    idx = 0
    print(f"[START] {in_path.name}  (in_fps≈{in_fps:.2f}, sample step={step})")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue

        if mirror_input:
            frame_bgr = cv2.flip(frame_bgr, 1)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = holistic.process(frame_rgb)

        frame_dict = {
            "left_hand":  lm_to_dict_list(res.left_hand_landmarks,  HAND_NAMES, include_visibility=False),
            "right_hand": lm_to_dict_list(res.right_hand_landmarks, HAND_NAMES, include_visibility=False),
            "pose":       lm_to_dict_list(res.pose_landmarks,       POSE_NAMES, include_visibility=True),
        }
        frames.append(frame_dict)

        if preview:
            # 간단 미리보기(양손/포즈 포인트 점만)
            draw = frame_bgr.copy()
            def dot(norm_x, norm_y, color):
                h, w = draw.shape[:2]
                x = int(norm_x * w); y = int(norm_y * h)
                cv2.circle(draw, (x, y), 2, color, -1)
            for arr, color in [(frame_dict["left_hand"], (0,255,0)),
                               (frame_dict["right_hand"], (0,128,255)),
                               (frame_dict["pose"], (255,0,0))]:
                for p in arr:
                    dot(p["x"], p["y"], color)
            cv2.imshow("preview (q=quit)", draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        idx += 1

    cap.release()
    holistic.close()
    if preview:
        cv2.destroyAllWindows()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(frames, f, ensure_ascii=False)

    print(f"[DONE]  {in_path.name} -> {out_path}  (frames={len(frames)})")
    return True

def is_video_file(p: Path):
    return p.suffix.lower() in {".mp4",".mov",".avi",".mkv",".webm",".m4v"}

def main():
    ap = argparse.ArgumentParser(description="영상→(Hands21+Pose33) JSON 덤프")
    ap.add_argument("--video", help="입력 영상 파일 경로 (생략하면 파일 열기 대화상자)")
    ap.add_argument("--dir", help="이 폴더 안의 모든 영상 처리")
    ap.add_argument("--out", help="출력 경로(파일 또는 폴더). 폴더면 입력파일명+_coords.json로 저장")
    ap.add_argument("--fps_out", type=float, default=30.0, help="샘플링 FPS (기본 30)")
    ap.add_argument("--mirror_input", action="store_true", help="입력 좌우 반전(셀피/미러 영상일 때)")
    ap.add_argument("--preview", action="store_true", help="프리뷰 창 표시(q로 종료)")
    ap.add_argument("--overwrite", action="store_true", help="기존 파일 덮어쓰기")
    args = ap.parse_args()

    inputs = []

    # 1) --video(단일)
    if args.video:
        p = Path(args.video).expanduser().resolve()
        if p.is_file() and is_video_file(p):
            inputs.append(p)
        else:
            print(f"[ERR] 영상 파일이 아님: {p}")
            sys.exit(1)

    # 2) --dir(여러 개)
    if args.dir:
        d = Path(args.dir).expanduser().resolve()
        if not d.is_dir():
            print(f"[ERR] 폴더가 아님: {d}")
            sys.exit(1)
        for p in d.iterdir():
            if p.is_file() and is_video_file(p):
                inputs.append(p)

    # 3) 아무 입력도 없으면 파일 선택 대화상자
    if not inputs:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(title="영상 파일을 선택하세요",
                                              filetypes=[("Video", "*.mp4;*.mov;*.avi;*.mkv;*.webm;*.m4v")])
            if not path:
                print("[CANCEL] 선택 안 함")
                sys.exit(0)
            inputs.append(Path(path))
        except Exception as e:
            print("[ERR] 파일 대화상자를 열 수 없습니다. --video 또는 --dir 옵션을 사용하세요.")
            print(e); sys.exit(1)

    # 출력 경로 규칙
    out_arg = Path(args.out).expanduser().resolve() if args.out else None

    for vid in inputs:
        if out_arg:
            if out_arg.suffix.lower() == ".json":
                out_path = out_arg  # 명시적으로 파일 지정
            else:
                # out이 폴더면, 입력파일명 + _coords.json
                out_folder = out_arg if out_arg.is_dir() or not out_arg.suffix else out_arg.parent
                out_path = (out_folder / f"{vid.stem}_coords.json").resolve()
        else:
            # 기본: 입력과 같은 폴더에 _coords.json
            out_path = (vid.parent / f"{vid.stem}_coords.json").resolve()

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] 이미 존재: {out_path}  (덮어쓰려면 --overwrite)")
            continue

        process_one_video(vid, out_path, args.fps_out, args.mirror_input, args.preview)

if __name__ == "__main__":
    main()