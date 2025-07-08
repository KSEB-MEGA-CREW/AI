from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import openai, os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS 허용 (모든 origin, 개발 테스트용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/stt/")
async def speech_to_text(file: UploadFile = File(...)):
    # 임시로 저장
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())
    # Whisper API 호출
    with open(temp_filename, "rb") as f:
        resp = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    os.remove(temp_filename)
    return {"text": resp.text}

# uvicorn stt_api:app --reload
