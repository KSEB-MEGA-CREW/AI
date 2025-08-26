import json
import os
import re

# This function cleans a gloss ID by removing whitespace and replacing characters that are invalid in filenames.
# 이 함수는 공백을 제거하고 파일 이름에 사용할 수 없는 문자를 대체하여 gloss ID를 정리합니다.
def clean_gloss_id(gloss_id):
    # Remove all whitespace characters.
    # 모든 공백 문자를 제거합니다.
    cleaned_id = re.sub(r'\s+', '', gloss_id)
    # Replace characters that are invalid in filenames with an underscore.
    # 파일 이름에 사용할 수 없는 문자를 밑줄로 대체합니다.
    cleaned_id = re.sub(r'[:/?*<>|"]', '_', cleaned_id)
    return cleaned_id


# This function processes JSON files one by one using a generator, which is memory efficient.
# 이 함수는 제너레이터를 사용하여 JSON 파일을 하나씩 처리하므로 메모리 효율적입니다.
def process_files_generator(root_folder, output_path):
    # Initialize a counter for processed files.
    # 처리된 파일 수를 세는 카운터를 초기화합니다.
    files_processed_count = 0
    # Get the absolute path of the output file to avoid processing it.
    # 출력 파일의 절대 경로를 가져와 자신을 처리하는 것을 방지합니다.
    output_abs_path = os.path.abspath(output_path)

    # Open the output file in write mode with UTF-8 encoding.
    # 출력 파일을 쓰기 모드와 UTF-8 인코딩으로 엽니다.
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Walk through the directory tree starting from the root folder.
        # 루트 폴더부터 시작하여 디렉터리 트리를 순회합니다.
        for dirpath, _, filenames in os.walk(root_folder):
            # Iterate over each filename in the current directory.
            # 현재 디렉터리의 각 파일 이름을 반복합니다.
            for filename in filenames:
                # Check if the file has a .json extension.
                # 파일 확장자가 .json인지 확인합니다.
                if filename.endswith('.json'):
                    # Construct the full path to the file.
                    # 파일의 전체 경로를 구성합니다.
                    file_path = os.path.join(dirpath, filename)
                    
                    # Skip the output file itself.
                    # 출력 파일 자체는 건너뜁니다.
                    if os.path.abspath(file_path) == output_abs_path:
                        continue

                    print(f"{files_processed_count}번째 처리중 ...\n")
                    # Try to process the JSON file.
                    # JSON 파일 처리를 시도합니다.
                    try:
                        # Open the JSON file in read mode.
                        # JSON 파일을 읽기 모드로 엽니다.
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            # Load the JSON data from the file.
                            # 파일에서 JSON 데이터를 로드합니다.
                            data = json.load(infile)
                            # Extract the Korean text from the data.
                            # 데이터에서 한국어 텍스트를 추출합니다.
                            korean_text = data['krlgg_sntenc']['koreanText']
                            # Extract and clean the gloss IDs from the data.
                            # 데이터에서 gloss ID를 추출하고 정리합니다.
                            gloss_ids = [clean_gloss_id(gesture['gloss_id']) for gesture in data['sign_script']['sign_gestures_strong']]
                            
                            # Create a dictionary with the processed data.
                            # 처리된 데이터로 딕셔너리를 생성합니다.
                            processed_item = {
                                "koreanText": korean_text,
                                "gloss_id": gloss_ids
                            }
                            
                            # Convert the dictionary to a JSON string and write it to the output file.
                            # 딕셔너리를 JSON 문자열로 변환하여 출력 파일에 씁니다.
                            outfile.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                            # Increment the processed files counter.
                            # 처리된 파일 카운터를 증가시킵니다.
                            files_processed_count += 1
                    # Handle cases where a key is not found in the JSON data.
                    # JSON 데이터에서 키를 찾을 수 없는 경우를 처리합니다.
                    except KeyError as e:
                        # Print an error message if a key is missing.
                        # 키가 누락된 경우 오류 메시지를 출력합니다.
                        print(f"Key error in file {file_path}: Key '{e}' not found.")
                    # Handle other unexpected errors during file processing.
                    # 파일 처리 중 다른 예기치 않은 오류를 처리합니다.
                    except Exception as e:
                        # Print an error message for any other exceptions.
                        # 다른 예외에 대한 오류 메시지를 출력합니다.
                        print(f"An unknown error occurred while processing file {file_path}: {e}")
    
    # Print a confirmation message with the total number of processed files.
    # 처리된 총 파일 수와 함께 확인 메시지를 출력합니다.
    print(f"File '{output_path}' created successfully. Total {files_processed_count} files processed.")

# Main execution block.
# 메인 실행 블록입니다.
if __name__ == "__main__":
    # Set the script directory, handling cases where __file__ is not defined.
    # __file__이 정의되지 않은 경우를 처리하여 스크립트 디렉터리를 설정합니다.
    try:
        # Get the directory of the current script.
        # 현재 스크립트의 디렉터리를 가져옵니다.
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback to the current working directory if __file__ is not available.
        # __file__을 사용할 수 없는 경우 현재 작업 디렉터리로 대체합니다.
        script_dir = os.getcwd()

    # Define the root folder containing the JSON files.
    # JSON 파일이 포함된 루트 폴더를 정의합니다.
    root_folder = os.path.abspath(os.path.join(script_dir, '..', '..', 'KSEB_json_data'))
    # Define the path for the output file.
    # 출력 파일의 경로를 정의합니다.
    output_path = os.path.join(script_dir, '..', 'preprocessed_data', 'processed_data.jsonl')

    # Try to run the file processing function.
    # 파일 처리 함수 실행을 시도합니다.
    try:
        # Call the function to process the files.
        # 파일을 처리하는 함수를 호출합니다.
        process_files_generator(root_folder, output_path)
    # Handle errors related to file I/O.
    # 파일 입출력과 관련된 오류를 처리합니다.
    except IOError as e:
        # Print an error message if the output file cannot be opened.
        # 출력 파일을 열 수 없는 경우 오류 메시지를 출력합니다.
        print(f"Error opening output file {output_path}: {e}")