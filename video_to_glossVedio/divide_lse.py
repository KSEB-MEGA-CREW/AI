import os
import json
from moviepy.video.io.VideoFileClip import VideoFileClip

base_dir = os.path.dirname(os.path.abspath(__file__))
strong_dir = os.path.join(base_dir, "strong")
weak_dir = os.path.join(base_dir, "weak")
json_files = [f for f in os.listdir(base_dir) if f.endswith(".json")]

for json_file in json_files:
    print(f"Processing {json_file}")
    json_path = os.path.join(base_dir, json_file)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_prefix = os.path.splitext(json_file)[0]
    video_path = os.path.join(base_dir, f"{file_prefix}.mp4")

    if not os.path.exists(video_path):
        print(f"  Video file not found: {video_path}")
        continue

    print(f"  Found video: {video_path}")
    video = VideoFileClip(video_path)

    # strong 폴더 생성 (없으면)
    strong_case_dir = os.path.join(strong_dir, file_prefix)
    os.makedirs(strong_case_dir, exist_ok=True)
    # weak 폴더 생성 (없으면)
    weak_case_dir = os.path.join(weak_dir, file_prefix)
    os.makedirs(weak_case_dir, exist_ok=True)

    # 왼쪽(강한 동작)
    for clip_info in data['sign_script'].get('sign_gestures_strong', []):
        start = clip_info['start']
        end = clip_info['end']
        gloss_id = clip_info['gloss_id']

        out_name = f"{file_prefix}_{gloss_id}.mp4"
        out_path = os.path.join(strong_case_dir, out_name)
        print(f"    Cutting(STRONG): {start}-{end} to {out_path}")
        try:
            subclip = video.subclip(start, end)
            subclip.write_videofile(out_path, codec="libx264", audio=False, verbose=False, logger=None)
            subclip.close()
        except Exception as e:
            print(f"    Error while saving {out_name}: {e}")

    # 오른쪽(약한 동작)
    for clip_info in data['sign_script'].get('sign_gestures_weak', []):
        start = clip_info['start']
        end = clip_info['end']
        gloss_id = clip_info['gloss_id']

        out_name = f"{file_prefix}_{gloss_id}.mp4"
        out_path = os.path.join(weak_case_dir, out_name)
        print(f"    Cutting(WEAK): {start}-{end} to {out_path}")
        try:
            subclip = video.subclip(start, end)
            subclip.write_videofile(out_path, codec="libx264", audio=False, verbose=False, logger=None)
            subclip.close()
        except Exception as e:
            print(f"    Error while saving {out_name}: {e}")

    video.close()
