FROM python:3.9

WORKDIR /app

# 라이브러리 설치
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드/모델 복사
COPY app/ .
COPY model/ ../model/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.9

WORKDIR /app

# 라이브러리 설치
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드/모델 복사
COPY app/ .
COPY model/ ../model/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]