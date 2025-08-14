from flask import Flask, request, jsonify
from transformers import pipeline
import download_from_s3
import config
import inference
# Flask 애플리케이션 객체 생성
app = Flask(__name__)

LOCAL_MODEL_PATH = config.LOCAL_MODEL_PATH
S3_MODEL_PATH = config.S3_MODEL_PATH
S3_UNI_GLOSS_SET = config.S3_UNI_GLOSS_SET
LOCAL_UNI_GLOSS_SET_PATH = config.LOCAL_UNI_GLOSS_SET_PATH

try:
    # Download the model from S3 to the local path
    download_from_s3.download_model_from_s3(S3_MODEL_PATH,  S3_UNI_GLOSS_SET, LOCAL_MODEL_PATH, LOCAL_UNI_GLOSS_SET_PATH)
    
except Exception as e:
        print(f"Error: 에러 메시지를 확인해주세요 {e}")

text_to_gloss_pipeline = pipeline(
        "translation_ko_to_KSL",  # This task is suitable for sequence-to-sequence models.
        model=LOCAL_MODEL_PATH,
        tokenizer=LOCAL_MODEL_PATH,
        device_map="auto"
    )

# URL 경로 '/'에 접속했을 때 실행될 함수 정의
@app.route('/', methods=['GET'])
def root_test():
    return jsonify({"value": "hihi"}), 200


@app.route('/T2G/translate', methods=['GET'])
def translate_text():

    """
    URL 파라미터로 텍스트를 받아 번역 결과를 JSON으로 반환합니다.
    예시 요청: http://127.0.0.1:5000/translate?text=Hello world
    """
    # 2. GET 요청에서 'text' 파라미터 추출
    text_to_translate = request.args.get('text')

    # # 'text' 파라미터가 없는 경우 에러 처리
    if not text_to_translate:
        return jsonify({"error": "번역할 'text' 파라미터가 필요합니다."}), 400

    # # 3. 로드된 모델로 예측 수행
    try:
        # 전역 변수로 로드된 translator를 사용해 예측 gloss list 출력
        prediction_gloss_list = inference.inference(text_to_translate, text_to_gloss_pipeline)
        
        
    #     # 4. 예측 결과를 JSON 형태로 반환
        return jsonify({"original_text": prediction_gloss_list, "translated_text": "hihi"})

    except Exception as e:
        # 모델 예측 중 에러가 발생할 경우
        print(f"Error during translation: {e}")
        return jsonify({"error": "번역 중 오류가 발생했습니다."}), 500




    # return jsonify({"value": "hihi"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1958, debug=True)

#     from flask import Flask, request, jsonify
# # from transformers import pipeline # 예시: 허깅페이스의 번역 모델을 사용할 경우

# # 1. 애플리케이션 시작 시 모델 로드
# # ==============================================================================
# # 여기에 실제 사용할 번역 모델 로드 코드를 작성합니다.
# # 예시: 허깅페이스의 영한 번역 모델
# # translator = pipeline("translation_en_to_ko", model="Helsinki-NLP/opus-mt-en-ko")

# # 실제 모델이 없을 경우를 대비한 가상 번역 함수 (테스트용)
# def dummy_translate(text):
#     """실제 번역 모델을 대체하는 가상 함수"""
#     print(f"번역할 텍스트: '{text}'")
#     # 실제로는 model.predict(text) 와 같은 코드가 위치합니다.
#     return f"Translated: {text}"

# # 전역 변수로 모델(또는 가상 함수)을 할당합니다.
# translator = dummy_translate 
# print(" * 번역 모델이 준비되었습니다.")
# # ==============================================================================


# app = Flask(__name__)

# @app.route('/translate', methods=['GET'])
# def translate_text():
#     """
#     URL 파라미터로 텍스트를 받아 번역 결과를 JSON으로 반환합니다.
#     예시 요청: http://127.0.0.1:5000/translate?text=Hello world
#     """
#     # 2. GET 요청에서 'text' 파라미터 추출
#     text_to_translate = request.args.get('text')

#     # 'text' 파라미터가 없는 경우 에러 처리
#     if not text_to_translate:
#         return jsonify({"error": "번역할 'text' 파라미터가 필요합니다."}), 400

#     # 3. 로드된 모델로 예측 수행
#     try:
#         # 전역 변수로 로드된 translator를 사용해 예측
#         prediction = translator(text_to_translate)
        
#         # 4. 예측 결과를 JSON 형태로 반환
#         return jsonify({"original_text": text_to_translate, "translated_text": prediction})

#     except Exception as e:
#         # 모델 예측 중 에러가 발생할 경우
#         print(f"Error during translation: {e}")
#         return jsonify({"error": "번역 중 오류가 발생했습니다."}), 500


# if __name__ == '__main__':
#     # 외부에서 접근 가능하도록 host='0.0.0.0' 설정
#     app.run(host='0.0.0.0', port=5000, debug=True)