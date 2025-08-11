import json
import random

def load_jsonl(file_path, limit=None, seed=42):
    '''
    Functions to load the data needed for training.
    If a limit is specified, it randomly samples that number of lines from the file.
    '''
    # 데이터를 저장할 빈 리스트를 생성합니다.
    data = []
    # 'with open'을 사용하여 파일을 열고, 작업이 끝나면 자동으로 파일을 닫도록 합니다.
    # 'encoding='utf-8''은 한글이 깨지지 않도록 인코딩 방식을 지정합니다.
    with open(file_path, 'r', encoding='utf-8') as f:
        # 파일의 모든 줄을 읽어 파이썬 딕셔너리로 변환합니다.
        for line in f:
            data.append(json.loads(line))

    # 만약 limit이 설정되어 있고, 전체 데이터의 크기가 limit보다 크다면,
    if limit is not None and len(data) > limit:
        random.seed(seed)
        # 전체 데이터에서 무작위로 limit 개수만큼 샘플링합니다.
        # random.sample은 비복원 추출을 수행하여 중복 없이 데이터를 선택합니다.
        data = random.sample(data, limit)

    # 데이터가 모두 담긴 리스트를 반환합니다.
    return data

def load_valid_gloss_set():
    # 유효한 글로스 목록 경로
    unique_glosses_path = '/home/202044005/KSEB/text_to_word/preprocessed_data/unique_glosses.json'
    try:
        with open(unique_glosses_path, 'r', encoding='utf-8') as f:
            valid_glosses = set(json.load(f))
    except FileNotFoundError:
        print(f"Error: 글로스 리스트 파일을 읽어올 수 없습니다. \n경로를 확인해주세요: {unique_glosses_path}")
        return None
    return valid_glosses

