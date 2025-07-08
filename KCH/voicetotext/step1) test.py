import openai
import os
from dotenv import load_dotenv

# 1. .env 파일 불러오기
load_dotenv()

# 2. 환경 변수에서 API 키 읽기
api_key = os.getenv("OPENAI_API_KEY")

# 3. OpenAI API 키 설정
openai.api_key = api_key

# 4. 변환하고 싶은 m4a 파일 경로
audio_file_path = "test.m4a"

# 5. 파일 열고 Whisper API 호출
with open(audio_file_path, "rb") as audio_file:
    transcript = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

print("음성 인식 결과:")
print(transcript.text)