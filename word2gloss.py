
import os
import json
import argparse

def find_word_in_json(word, root_dir):
    """
    지정된 디렉토리와 그 하위 디렉토리에서 주어진 단어를 포함하는 JSON 파일을 찾습니다.

    Args:
        word (str): 찾고자 하는 단어.
        root_dir (str): 검색을 시작할 최상위 디렉토리.

    Returns:
        list: 단어를 포함하는 JSON 파일의 경로 리스트.
    """
    found_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 'krlgg_sntenc' 객체 안의 'koreanText' 필드 확인
                        if 'krlgg_sntenc' in data and 'koreanText' in data['krlgg_sntenc'] and word in data['krlgg_sntenc']['koreanText']:
                            found_files.append(filepath)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing file {filepath}: {e}")
    return found_files

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Search for a word in JSON files.')
    # parser.add_argument('word', type=str, help='The word to search for.')
    # parser.add_argument('--dir', type=str, default='data', help='The directory to search in.')
    # args = parser.parse_args()

    # 찾고 싶은 단어를 여기에 직접 입력하세요.
    search_word = "입니다"
    search_dir = "data" # 또는 원하는 디렉토리 경로

    found_files = find_word_in_json(search_word, search_dir)

    if found_files:
        print(f"Found '{search_word}' in the following files:")
        for file in found_files:
            print(file)
    else:
        print(f"Could not find '{search_word}' in any JSON files.")
