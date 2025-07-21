# 'json' 라이브러리를 임포트합니다.
import json
# 'torch' 라이브러리를 임포트합니다.
import torch
# 'transformers' 라이브러리에서 필요한 클래스들을 임포트합니다.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# 'datasets' 라이브러리에서 'Dataset' 클래스를 임포트합니다.
from datasets import Dataset
# 사용자의 데이터 로드 함수를 임포트합니다.
import data_load
import config
import os

# 1. 기본 변수 및 데이터 로드
# 원본 사전 학습 모델
MODEL_NAME = config.MODEL_NAME
# 학습 데이터 경로
DATA_PATH = config.DATA_PATH
# 새 토크나이저와 모델을 저장할 디렉토리
NEW_MODEL_DIR = config.NEW_MODEL_DIR

# 전체 데이터를 로드합니다 (토큰 추출을 위해 limit 없이).
full_data = data_load.load_jsonl(DATA_PATH)
# print(type(full_data))

# --- 여기서부터 핵심 로직 ---

def extract_unique_gloss(data = full_data):
    """
    Functions to extract unique gloss for new data

    Args:
        data (list): JSONL data to add to the talkize
        
    Returns:
        unique_gloss_tokens (list): list of unique tokens extracted from data
        
    (KOREAN) 

    새로운 data에 대해 고유한 gloss를 추출하는 함수

    매게변수:
        data (list)= 토크나이저에 추가할 JSONL data

    반환값:
        unique_gloss_tokens (list)= data에서 추출한 고유 토큰 list
    """
    print("데이터셋에서 고유한 gloss_id를 추출합니다...")
    all_glosses = set()
    for item in data:
        all_glosses.update(item['gloss_id'])
    unique_gloss_tokens = list(all_glosses)
    print(f"총 {len(unique_gloss_tokens)}개의 고유한 gloss를 찾았습니다.")
    print(f"샘플 gloss: {unique_gloss_tokens[:10]}") # 처음 10개만 샘플로 출력
    return unique_gloss_tokens

def  adding_new_token(MODEL_NAME = MODEL_NAME, data = full_data):
    '''
    Function to import a model and add a new token to the talkizer

    Args:
        MODEL_NAME (str): Name of the model to add the token to
        data (list): Data for the token to add

    ###########################################

    (korean)

    모델을 불러와 토크나이저에 새로운 토큰을 추가하는 함수
    
    매게변수 :
        MODEL_NAME (str) = 토큰을 추가할 모델의 이름
        data (list) = 추가할 토큰에 대한 데이터
    '''
    print(f"'{MODEL_NAME}'에서 기존 토크나이저와 모델을 로드합니다.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 토큰 추가 전 어휘 사전 크기 확인
    original_vocab_size = len(tokenizer)
    print(f"토큰 추가 전 어휘 사전 크기: {original_vocab_size}")

    # 4. 토크나이저에 새로운 토큰 추가
    print("토크나이저에 새로운 gloss 토큰을 추가합니다...")
    unique_gloss_tokens = extract_unique_gloss(data)
    if unique_gloss_tokens == 0 : return 

    # .add_tokens()는 실제로 어휘 사전에 추가된 새로운 토큰의 수를 반환합니다.
    num_added_toks = tokenizer.add_tokens(unique_gloss_tokens)
    print(f"총 {num_added_toks}개의 새로운 토큰이 추가되었습니다.")

    # 토큰 추가 후 어휘 사전 크기 확인
    new_vocab_size = len(tokenizer)
    print(f"토큰 추가 후 어휘 사전 크기: {new_vocab_size}")

    # 5. 모델의 임베딩 레이어 크기 조절 (⭐ 매우 중요)
    # 토크나이저의 어휘 사전이 늘어났으므로, 모델의 토큰 임베딩 레이어 크기도 맞춰주어야 합니다.
    # 그렇지 않으면 학습 시 에러가 발생합니다.
    print("모델의 토큰 임베딩 레이어 크기를 조절합니다...")
    model.resize_token_embeddings(new_vocab_size)

    # 6. 업데이트된 토크나이저와 모델 저장
    print(f"업데이트된 토크나이저와 모델을 '{NEW_MODEL_DIR}'에 저장합니다.")
    tokenizer.save_pretrained(NEW_MODEL_DIR)
    model.save_pretrained(NEW_MODEL_DIR)
    print("저장 완료!")

# def load_model():
#     print("\n--- 저장된 새 토크나이저와 모델 로드 테스트 ---")
#     loaded_model = AutoModelForSeq2SeqLM.from_pretrained(NEW_MODEL_DIR)
#     return loaded_model

def load_model(model_dir=NEW_MODEL_DIR, base_model_name=MODEL_NAME):
    """
    If a model exists in the specified directory (model_dir), it will load that model and torqueizer; 
    if  does not exist, it will load the default model (base_model_name) from the Hugging Face hub.
    
    Args:
        model_dir (str): Locally saved model and torquenizer paths
        base_model_name (str): Default model name for the Hugging Face hub

    Returns:
        tuple: (loaded models, loaded torquers)

    
    (korean)
    
    지정된 디렉토리(model_dir)에 모델이 존재하면 해당 모델과 토크나이저를 로드하고,
    존재하지 않으면 Hugging Face 허브에서 기본 모델(base_model_name)을 로드합니다.

    매게변수:
        model_dir (str): 로컬에 저장된 모델 및 토크나이저 경로
        base_model_name (str): Hugging Face 허브의 기본 모델 이름

    반환값:
        tuple: (로드된 모델, 로드된 토크나이저)
    """
    # 지정된 디렉토리가 존재하고, 내부에 파일이 있는지 확인합니다.
    if os.path.exists(model_dir) and os.listdir(model_dir):
        # 디렉토리가 존재하면, 로컬에 저장된 모델과 토크나이저를 로드합니다.
        print(f"'{model_dir}'에서 로컬 모델과 토크나이저를 로드합니다.")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        # 디렉토리가 없거나 비어있으면, Hugging Face 허브에서 기본 모델을 로드합니다.
        print(f"'{model_dir}'에 로컬 모델이 없습니다. Hugging Face 허브에서 '{base_model_name}'을(를) 로드합니다.")
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # 로드된 모델과 토크나이저를 반환합니다.
    return model, tokenizer




# # 2. 데이터셋에서 모든 고유한 gloss_id 추출
# print("데이터셋에서 고유한 gloss_id를 추출합니다...")
# all_glosses = set()
# for item in full_data:
#     # item['gloss_id']가 리스트 형태이므로, set에 각 요소를 추가
#     all_glosses.update(item['gloss_id'])

# # 집합(set)을 다시 리스트로 변환
# unique_gloss_tokens = list(all_glosses)
# print(f"총 {len(unique_gloss_tokens)}개의 고유한 gloss를 찾았습니다.")
# print(f"샘플 gloss: {unique_gloss_tokens[:10]}") # 처음 10개만 샘플로 출력

# # 3. 기존 토크나이저와 모델 로드
# print(f"'{MODEL_NAME}'에서 기존 토크나이저와 모델을 로드합니다.")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# # 토큰 추가 전 어휘 사전 크기 확인
# original_vocab_size = len(tokenizer)
# print(f"토큰 추가 전 어휘 사전 크기: {original_vocab_size}")

# # 4. 토크나이저에 새로운 토큰 추가
# print("토크나이저에 새로운 gloss 토큰을 추가합니다...")
# # .add_tokens()는 실제로 어휘 사전에 추가된 새로운 토큰의 수를 반환합니다.
# num_added_toks = tokenizer.add_tokens(unique_gloss_tokens)
# print(f"총 {num_added_toks}개의 새로운 토큰이 추가되었습니다.")

# # 토큰 추가 후 어휘 사전 크기 확인
# new_vocab_size = len(tokenizer)
# print(f"토큰 추가 후 어휘 사전 크기: {new_vocab_size}")

# # 5. 모델의 임베딩 레이어 크기 조절 (⭐ 매우 중요)
# # 토크나이저의 어휘 사전이 늘어났으므로, 모델의 토큰 임베딩 레이어 크기도 맞춰주어야 합니다.
# # 그렇지 않으면 학습 시 에러가 발생합니다.
# print("모델의 토큰 임베딩 레이어 크기를 조절합니다...")
# model.resize_token_embeddings(new_vocab_size)

# # 6. 업데이트된 토크나이저와 모델 저장
# print(f"업데이트된 토크나이저와 모델을 '{NEW_MODEL_DIR}'에 저장합니다.")
# tokenizer.save_pretrained(NEW_MODEL_DIR)
# model.save_pretrained(NEW_MODEL_DIR)
# print("저장 완료!")


adding_new_token()

# 7. 저장된 토크나이저와 모델 다시 로드하여 확인
print("\n--- 저장된 새 토크나이저와 모델 로드 테스트 ---")

loaded_model, loaded_tokenizer = load_model()
# loaded_tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL_DIR)
# loaded_model = AutoModelForSeq2SeqLM.from_pretrained(NEW_MODEL_DIR)

print(f"로드된 토크나이저의 어휘 사전 크기: {len(loaded_tokenizer)}")

# 추가된 토큰이 잘 처리되는지 테스트
test_gloss =  extract_unique_gloss()[0] # 아까 찾은 고유 gloss 중 하나로 테스트
encoded = loaded_tokenizer.encode(test_gloss)
decoded = loaded_tokenizer.decode(encoded)

print(f"테스트 Gloss: '{test_gloss}'")
# [0, 50001, 2] 와 같이 시작/끝 특수 토큰과 함께 단일 ID로 인코딩되어야 합니다.
print(f"인코딩 결과 (토큰 ID): {encoded}")
# 디코딩 결과는 원본 gloss와 같아야 합니다.
print(f"디코딩 결과: '{decoded}'")