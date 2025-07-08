import openai
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import webrtcvad
import queue
import threading
import sys
import time
from dotenv import load_dotenv
import tempfile

# 환경변수 불러오기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30    # ms, webrtcvad 권장: 10, 20, 30
MAX_SILENCE_SEC = 2    # 무음 2초 이상이면 자동 종료

def record_with_vad():
    vad = webrtcvad.Vad(2)  # 0~3(엄격), 2는 중간 정도
    q = queue.Queue()
    stop_record = threading.Event()
    audio = []

    def callback(indata, frames, time_, status):
        q.put(indata.copy())

    def record_thread():
        print("\n말씀하세요. (무음 2초 자동 종료)")
        silence_chunks = 0
        started = False

        with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, dtype='int16', callback=callback, blocksize=int(SAMPLE_RATE * FRAME_DURATION / 1000)):
            while not stop_record.is_set():
                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                # vad가 처리할 프레임 단위로 reshape
                is_speech = vad.is_speech(chunk.tobytes(), SAMPLE_RATE)
                if is_speech:
                    silence_chunks = 0
                    started = True
                else:
                    if started:
                        silence_chunks += 1
                audio.append(chunk)
                # 무음 2초면 자동 종료
                if started and silence_chunks * (FRAME_DURATION/1000) > MAX_SILENCE_SEC:
                    stop_record.set()
                    break

    t = threading.Thread(target=record_thread)
    t.start()
    t.join()

    # 전체 음성 데이터 합치기
    full_audio = np.concatenate(audio, axis=0)
    return full_audio

def main():
    print("=== STT 녹음 ===")
    print("녹음을 시작하려면 s 키를 누르세요 (종료: Ctrl+C)")
    try:
        while True:
            key = input("\n[s 입력 후 Enter] : ")
            if key.strip().lower() != 's':
                continue

            audio_data = record_with_vad()

            # 임시 wav 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(tmpfile.name, audio_data, SAMPLE_RATE)
                tmp_filename = tmpfile.name

            print("Whisper 변환 중...")
            with open(tmp_filename, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            print("변환 결과:", transcript.text)
            os.remove(tmp_filename)
    except KeyboardInterrupt:
        print("\n프로그램 종료.")

if __name__ == "__main__":
    main()