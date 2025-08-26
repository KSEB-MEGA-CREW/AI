
import json
import re
import os

def clean_data(input_file_path, output_file_path):
    """
    Cleans the processed_data.jsonl file based on several criteria.
    여러 기준에 따라 processed_data.jsonl 파일을 정제합니다.

    Args:
        input_file_path (str): The path to the input JSONL file. (입력 JSONL 파일 경로)
        output_file_path (str): The path to save the cleaned JSONL file. (정제된 JSONL 파일을 저장할 경로)
    """
    # Define cleaning parameters
    # 정제 파라미터 정의
    MIN_LEN = 3  # 최소 단어 길이 (Minimum word length)
    MAX_LEN = 50  # 최대 단어 길이 (Maximum word length)
    LEN_RATIO_THRESHOLD = 1.6  # 길이 비율 임계값 (Length ratio threshold)

    cleaned_data = []
    removed_count = 0

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                korean_text = data.get("koreanText", "")
                gloss_ids = data.get("gloss_id", [])

                # --- 1. 정규화 (Normalization) ---
                # 여러 개의 공백을 하나로 변경
                # Replace multiple spaces with a single space
                korean_text = re.sub(r'\s+', ' ', korean_text).strip()
                
                # 간단한 문장 부호 처리 (마침표, 쉼표 주변에 공백 추가)
                # Simple punctuation handling (add space around periods, commas)
                korean_text = re.sub(r'([,.!?])', r' \1 ', korean_text)
                korean_text = re.sub(r'\s+', ' ', korean_text).strip()


                # --- 2. 필터링 (Filtering) ---
                korean_len = len(korean_text.split())
                gloss_len = len(gloss_ids)

                # 빈 데이터 필터링
                # Filter empty data
                if not korean_text or not gloss_ids:
                    removed_count += 1
                    continue

                # # 길이 기반 필터링
                # # Length-based filtering
                # if not (MIN_LEN <= korean_len <= MAX_LEN and MIN_LEN <= gloss_len <= MAX_LEN):
                #     removed_count += 1
                #     continue
                
                # 길이 비율 필터링
                # Length ratio filtering
                if korean_len > 0 and gloss_len > 0:
                    ratio = max(korean_len, gloss_len) / min(korean_len, gloss_len)
                    if ratio > LEN_RATIO_THRESHOLD:
                        removed_count += 1
                        continue

                # 내용 기반 필터링 (한글 비율 확인)
                # Content-based filtering (check Korean character ratio)
                korean_chars = re.findall(r'[가-힣]', korean_text)
                if len(korean_chars) / len(korean_text.replace(" ", "")) < 0.8:
                    removed_count += 1
                    continue
                
                # 정제된 데이터 추가
                # Add cleaned data
                data['koreanText'] = korean_text
                cleaned_data.append(data)

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line {i+1}")
                removed_count += 1
                continue

    # 정제된 데이터를 새 파일에 저장
    # Save cleaned data to a new file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"데이터 정제 완료 (Data cleaning complete)")
    print(f"총 {len(cleaned_data) + removed_count}개 데이터 중 {removed_count}개 제거됨")
    print(f"Total {removed_count} items removed out of {len(cleaned_data) + removed_count}")
    print(f"정제된 데이터가 '{output_file_path}'에 저장되었습니다.")
    print(f"Cleaned data saved to '{output_file_path}'.")


if __name__ == '__main__':
    # Get the absolute path of the current script
    # 현재 스크립트의 절대 경로를 가져옵니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the input file path relative to the script's location
    # 스크립트 위치를 기준으로 입력 파일 경로를 구성합니다.
    input_path = os.path.join(os.path.dirname(current_dir), 'preprocessed_data', 'processed_data.jsonl')
    # Construct the output file path
    # 출력 파일 경로를 구성합니다.
    output_path = os.path.join(os.path.dirname(current_dir), 'preprocessed_data', 'cleaned_data.jsonl')
    
    clean_data(input_path, output_path)
