from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

img_path = 'image/test.jpg'  # 테스트할 이미지 파일 경로
img = cv2.imread(img_path)

results = model.predict(img, conf=0.5, verbose=False)
annotated_img = results[0].plot()

cv2.imshow("YOLOv8 Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()