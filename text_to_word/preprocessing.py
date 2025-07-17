# import json
# import glob
# import os

# # 실행 위치에 관계없이 경로를 올바르게 설정합니다.
# # 스크립트 파일의 위치를 기준으로 절대 경로를 생성합니다.
# script_dir = os.path.dirname(os.path.abspath(__file__))
# # root_folder = os.path.abspath(os.path.join(script_dir, '..', 'KSEB_json_data', 'NIKL_Sign Language Parallel Corpus_2024_BD_PP', '2024_0586690-0606850_1988_BD_PP1'))
# root_folder = os.path.abspath(os.path.join(script_dir, '..', 'KSEB_json_data'))
# output_path = os.path.join(script_dir, 'processed_data.json')

# # List to hold the extracted data
# output_data = []

# # Utilizes an OS library to recursively find .json files in a specified folder and all subfolders.
# file_pattern = os.path.join(root_folder, '**', '*.json')
# json_files = glob.glob(file_pattern, recursive=True)
# print(file_pattern)
# print(json_files)

# # Exclude the output file (processed_data.json) from being included in the processing target list.
# # os.path.abspath를 사용하여 절대 경로를 비교합니다.
# output_abs_path = os.path.abspath(output_path)
# json_files_to_process = [f for f in json_files if os.path.abspath(f) != output_abs_path]


# # Traverse each JSON file and extract the data.
# for file_path in json_files_to_process:
#     try:

#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             korean_text = data['krlgg_sntenc']['koreanText']
#             gloss_ids = [gesture['gloss_id'] for gesture in data['sign_script']['sign_gestures_strong']]
#             output_data.append({
#                 "koreanText": korean_text,
#                 "gloss_id": gloss_ids
#             })
#     except Exception as e:
#         print(f"파일 처리 중 오류 발생 {file_path}: {e}")

# # Write the combined data to the output file
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(output_data, f, ensure_ascii=False, indent=2)

# print(f"'{output_path}' 파일이 생성되었습니다. 총 {len(json_files_to_process)}개의 파일이 처리되었습니다.")


import json
import glob
import os

# Set the path correctly regardless of where it is executed.
# 실행 위치에 관계없이 경로를 올바르게 설정합니다.
# Generate absolute paths based on the location of the script file.
# 스크립트 파일의 위치를 기준으로 절대 경로를 생성합니다.
# Add exception handling for environments where __file__ is undefined (e.g. Jupyter notebooks).
# __file__이 정의되지 않은 환경(예: Jupyter 노트북)을 위해 예외 처리를 추가합니다.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

root_folder = os.path.abspath(os.path.join(script_dir, '..', 'KSEB_json_data'))
# Output file path
output_path = os.path.join(script_dir, 'processed_data.jsonl')

# Utilizes an OS library to recursively find .json files in a specified folder and all subfolders.
file_pattern = os.path.join(root_folder, '**', '*.json')
json_files = glob.glob(file_pattern, recursive=True)

print(file_pattern)
print(json_files)

# Exclude the output file from being included in the processing target list.
# os.path.abspath를 사용하여 절대 경로를 비교합니다.
output_abs_path = os.path.abspath(output_path)
json_files_to_process = [f for f in json_files if os.path.abspath(f) != output_abs_path]

files_processed_count = 0

# Open the file in write mode, and write the extracted data from each JSON file line by line.
# 파일을 쓰기 모드로 열고, 각 JSON 파일에서 추출한 데이터를 한 줄씩 씁니다.
try:
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Traverse each JSON file and extract the data.
        for file_path in json_files_to_process:
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    data = json.load(infile)
                    korean_text = data['krlgg_sntenc']['koreanText']
                    gloss_ids = [gesture['gloss_id'] for gesture in data['sign_script']['sign_gestures_strong']]
                    
                    # 처리할 데이터를 딕셔너리로 만듭니다.
                    processed_item = {
                        "koreanText": korean_text,
                        "gloss_id": gloss_ids
                    }
                    
                    # json.dumps를 사용하여 딕셔너리를 JSON 문자열로 변환하고 파일에 씁니다.
                    # 각 줄이 하나의 JSON 객체가 되도록 줄바꿈 문자를 추가합니다.
                    outfile.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                    files_processed_count += 1

            except KeyError as e:
                print(f"파일 처리 중 키 오류 발생 {file_path}: 키 '{e}'를 찾을 수 없습니다.")
            except Exception as e:
                print(f"파일 처리 중 알 수 없는 오류 발생 {file_path}: {e}")

    print(f"'{output_path}' 파일이 생성되었습니다. 총 {files_processed_count}개의 파일이 처리되었습니다.")

except IOError as e:
    print(f"출력 파일 열기 오류 {output_path}: {e}")
