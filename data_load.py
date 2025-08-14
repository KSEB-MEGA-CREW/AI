import json

def load_valid_gloss_set():
    # 유효한 글로스 목록 경로
    # unique_glosses_path = './T2G_flask/preprocessed_data/unique_glosses.json'
    unique_glosses_path = './preprocessed_data/unique_glosses.json'
    try:
        with open(unique_glosses_path, 'r', encoding='utf-8') as f:
            valid_glosses = set(json.load(f))
    except FileNotFoundError:
        print(f"Error: 글로스 리스트 파일을 읽어올 수 없습니다. \n경로를 확인해주세요: {unique_glosses_path}")
        return None
    return valid_glosses

