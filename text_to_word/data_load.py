import json

def load_jsonl(file_path, limit = None):
        '''
        Functions to load the data needed for training
        '''
        # 데이터를 저장할 빈 리스트를 생성합니다.
        data = []
        # 'with open'을 사용하여 파일을 열고, 작업이 끝나면 자동으로 파일을 닫도록 합니다.
        # 'encoding='utf-8''은 한글이 깨지지 않도록 인코딩 방식을 지정합니다.
        with open(file_path, 'r', encoding='utf-8') as f:
            # 파일의 각 줄(line)에 대해 반복합니다.
            ##enumerater가 뭐지??????????????
            for i, line in enumerate(f):
                # 'json.loads(line)'을 사용하여 JSON 문자열 한 줄을 파이썬 딕셔너리로 변환하고, 'data' 리스트에 추가합니다.
                # if a limit is set and the current count has reached the limit
                # stop reading
                if limit is not None and i >= limit: break

                data.append(json.loads(line))
        # 데이터가 모두 담긴 리스트를 반환합니다.
        return data

