import cv2
import os

video_path = "sample.mp4"
save_dir = "images/hello"   # 한글 대신 영어!
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("⚠️ 영상 파일을 열 수 없습니다. 경로를 확인하세요.")
    exit()

count = 0
save_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("✅ 영상 끝 or 프레임 없음")
        break
    if count % 5 == 0:
        img_path = f"{save_dir}/frame_{count}.jpg"
        cv2.imwrite(img_path, frame)
        if os.path.exists(img_path):
            print(f"✔️ 저장 성공: {img_path}")
            save_count += 1
        else:
            print(f"❌ 저장 실패: {img_path}")
    count += 1
cap.release()

print(f"총 {save_count}장의 이미지가 저장되었습니다.")