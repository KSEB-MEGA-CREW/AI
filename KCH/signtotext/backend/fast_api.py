# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
from tensorflow.keras.models import load_model

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 혹은 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 & 라벨 로딩
model = load_model("model\gesture_model.h5")
with open("model\label_map.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    sequence = np.array(body).reshape(1, 12, 194)  # REQUIRED_FRAMES=12, EXPECTED_LEN=194

    prediction = model.predict(sequence)[0]
    max_index = int(np.argmax(prediction))
    confidence = float(prediction[max_index])
    label = labels[max_index] if confidence > 0.4 else "none"

    return { "label": label }