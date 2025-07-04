'''IMPORT'''
from inference import InferencePipeline                         #1) 영상 스트리밍과 객체 탐지 파이프라인을 관리해주는 클래스인 InferencePipeline을 import해줍니다.
from inference.core.interfaces.stream.sinks import render_boxes #2) 네모난 상자를 그려주는 함수를 import해줍니다.

#3) pipeline을 초기화 해줍니다. 여기서 pipeline은 파이프라인은 일련의 단계들을 하나로 묶어서 한 번에 처리하는 조립식 작업 흐름이라고 이해하면 됩니다.
#3) inferencepipeline의 init함수를 통해서 yolo모델을 불러오고, video_reference를 0으로 초기화해서 첫 번째 카메라 장치를 사용하게 하고, on_prediction=render_boxes를 통해서
#3) 탐지결과를 받아 화면에 박스를 그리도록 합니다.
pipeline = InferencePipeline.init(
    model_id="yolov8n-640",
    video_reference=0,
    on_prediction=render_boxes
)

