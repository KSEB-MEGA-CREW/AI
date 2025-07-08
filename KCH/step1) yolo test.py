from ultralytics import YOLO
import cv2

# YOLOv8n(가장 가벼운 사전학습 모델) 로드
model = YOLO('yolov8n.pt')

# 웹캠 열기 (0: 기본 카메라)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 예측 (결과는 results[0]에 들어 있음)
    results = model.predict(frame, conf=0.5, verbose=False)

    # 결과를 프레임 위에 그리기
    annotated_frame = results[0].plot()  # Bounding box, label 표시

    # 화면에 출력
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()