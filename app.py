from flask import Flask
# from transformers import pipeline
# Flask 애플리케이션 객체 생성
app = Flask(__name__)

# translator = pipeline()

# URL 경로 '/'에 접속했을 때 실행될 함수 정의
@app.route('/T2G/trasrate', methods=['GET'])
def translate_text():

    """
    URL 파라미터로 텍스트를 받아 번역 결과를 JSON으로 반환합니다.
    예시 요청: http://127.0.0.1:5000/translate?text=Hello world
    """
    # 2. GET 요청에서 'text' 파라미터 추출
    # text_to_translate = request.args.get('text')

    # # 'text' 파라미터가 없는 경우 에러 처리
    # if not text_to_translate:
    #     return jsonify({"error": "번역할 'text' 파라미터가 필요합니다."}), 400

    # # 3. 로드된 모델로 예측 수행
    # try:
    #     # 전역 변수로 로드된 translator를 사용해 예측
    #     prediction = translator(text_to_translate)
        
    #     # 4. 예측 결과를 JSON 형태로 반환
    #     return jsonify({"original_text": text_to_translate, "translated_text": prediction})

    # except Exception as e:
    #     # 모델 예측 중 에러가 발생할 경우
    #     print(f"Error during translation: {e}")
    #     return jsonify({"error": "번역 중 오류가 발생했습니다."}), 500




    return 'Hello, Flask!', 200

# 이 파일을 직접 실행했을 때 개발용 웹 서버를 구동
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)