import boto3
import config
import torch
import os
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# 'transformers' 라이브러리에서 필요한 클래스들을 임포트합니다.
from transformers import (
    # 'AutoTokenizer'는 사전 학습된 모델에 맞는 토크나이저를 자동으로 로드하는 클래스입니다.
    AutoTokenizer,
    # 'AutoModelForSeq2SeqLM'은 텍스트-투-텍스트(번역, 요약 등) 과업을 위한 사전 학습된 모델을 자동으로 로드하는 클래스입니다.
    AutoModelForSeq2SeqLM,
)
import data_load


#------------------------------------
# 0. 상수 초기화
#------------------------------------

# 입력 시퀀스(한국어 문장)의 최대 길이를 128 토큰으로 설정합니다. 이보다 길면 잘라냅니다.
max_input_length = 128
# 타겟 시퀀스(수어 단어)의 최대 길이를 128 토큰으로 설정합니다.
max_target_length = 128

S3_MODEL_PATH = config.S3_MODEL_PATH
S3_GLOSSES_SET_PATH = config.S3_GLOSSES_SET_PATH
S3_UNI_GLOSS_SET = config.S3_UNI_GLOSS_SET
LOCAL_MODEL_PATH = config.LOCAL_MODEL_PATH # 모델을 다운로드할 로컬 경로
LOCAL_UNI_GLOSS_SET_PATH = config.LOCAL_UNI_GLOSS_SET_PATH

# Helper function to download a single file from S3, with existence check
# 이미 있는 파일을 제외하고 파일을 다운로드 시키는 helper 함수
def _download_s3_object(s3_client, bucket_name, s3_object, local_dir):
    # Get the object key (file path in S3)
    key = s3_object['Key']
    # Get the size of the S3 object
    s3_object_size = s3_object['Size']
    # Construct the full local path for the file
    local_file_path = os.path.join(local_dir, os.path.basename(key))

    # Check if the file already exists locally
    # 파일이 local에 이미 존재하는지 확인
    if os.path.exists(local_file_path):
        # Get the size of the local file
        # 파일의 사이즈를 가져오기
        local_file_size = os.path.getsize(local_file_path)
        # If sizes match, skip downloading and return a message
        # 파일 사이즈가 다운로드 할 사이즈와 동일하다면 스킵
        if local_file_size == s3_object_size:
            # Return a message indicating the file is skipped
            return f"Skipping {key}, already exists and size matches."

    # Print a message indicating the download is starting
    print(f"Downloading {key} to {local_file_path}")
    # Use a try-except block for robust error handling
    try:
        # Download the file from S3 to the local path
        s3_client.download_file(bucket_name, key, local_file_path)
        # Return a success message
        return f"Successfully downloaded {key}"
    # Catch any exception during download
    except Exception as e:
        # Return an error message if download fails
        return f"Failed to download {key}: {e}"

# 모델과 uniqe gloss set을 다운로드
def download_model_from_s3(s3_model_path, s3_gloss_set_path, local_model_path, local_gloss_set_path):
    # Create the local directory for the model if it doesn't exist, ignoring errors if it already exists
    # 디렉터리가 없다면 만들기
    os.makedirs(local_model_path, exist_ok=True)
    # Get the directory path for the gloss set
    gloss_set_dir_path = os.path.dirname(local_gloss_set_path)
    # Create the local directory for the gloss set if it doesn't exist
    os.makedirs(gloss_set_dir_path, exist_ok=True)

    # Parse the S3 model path to get bucket and prefix
    parsed_model_url = urlparse(s3_model_path)
    # Get the bucket name from the parsed URL
    bucket_name = parsed_model_url.netloc
    # Get the S3 object prefix for the model, removing any leading slashes
    model_prefix = parsed_model_url.path.lstrip('/')

    # Parse the S3 gloss set path to get its prefix
    parsed_gloss_set_url = urlparse(s3_gloss_set_path)
    # Get the S3 object prefix for the gloss set
    gloss_set_prefix = parsed_gloss_set_url.path.lstrip('/')

    # Initialize the S3 client from boto3
    s3_client = boto3.client('s3')
    # Create a paginator to handle multiple pages of S3 objects
    paginator = s3_client.get_paginator('list_objects_v2')

    # --- Collect all file objects to download ---
    # Create a list to store all S3 objects and their target local directory
    objects_to_download = []
    
    # Paginate through model files
    print("--- Collecting model file list from S3 ---")
    # Get an iterator for the pages of model objects
    model_pages = paginator.paginate(Bucket=bucket_name, Prefix=model_prefix)
    # Iterate over each page of model objects
    for page in model_pages:
        # Check if the page contains any objects
        if "Contents" in page:
            # Iterate over each object on the page
            for obj in page['Contents']:
                # Add file to list if it's not a directory (i.e., does not end with '/')
                if not obj['Key'].endswith('/'):
                    # Append a tuple of the object and its destination path
                    objects_to_download.append((obj, local_model_path))

    # Paginate through gloss set files
    print("--- Collecting gloss set file list from S3 ---")
    # Get an iterator for the pages of gloss set objects
    gloss_pages = paginator.paginate(Bucket=bucket_name, Prefix=gloss_set_prefix)
    # Iterate over each page of gloss set objects
    for page in gloss_pages:
        # Check if the page contains any objects
        if "Contents" in page:
            # Iterate over each object on the page
            for obj in page['Contents']:
                # Add file to list if it's not a directory
                if not obj['Key'].endswith('/'):
                    # Append a tuple of the object and its destination path
                    objects_to_download.append((obj, gloss_set_dir_path))

    # --- Download files concurrently ---
    # Print a status message with the total number of files
    print(f"\n--- Starting concurrent download of {len(objects_to_download)} files ---")
    # Use ThreadPoolExecutor for parallel downloads, with a max of 10 workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create a dictionary to hold future objects, submitting each download task
        future_to_obj = {executor.submit(_download_s3_object, s3_client, bucket_name, obj, local_dir): obj for obj, local_dir in objects_to_download}
        # Iterate over completed futures as they finish
        for future in as_completed(future_to_obj):
            # Get the original object for the completed future
            obj = future_to_obj[future]
            # Use a try-except block to handle potential exceptions from the future
            try:
                # Get the result of the download operation (the return value from _download_s3_object)
                result = future.result()
                # Print the result
                print(result)
            # Catch any exception that occurred during the download
            except Exception as exc:
                # Print an error message including the key and the exception
                print(f'{obj['Key']} generated an exception: {exc}')



def pipline():
    try:
        # Download the model from S3 to the local path
        download_model_from_s3(S3_MODEL_PATH,  S3_UNI_GLOSS_SET, LOCAL_MODEL_PATH, LOCAL_UNI_GLOSS_SET_PATH)
        
        # Load the tokenizer and model from the local path
        trained_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        trained_model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH, device_map="auto", ignore_mismatched_sizes=True)
        
        # Load the gloss set and set the model to evaluation mode
        uni_glosses_set = data_load.load_valid_gloss_set()
        trained_model.eval()
        
        return trained_model, trained_tokenizer, uni_glosses_set
    except Exception as e:
        print(f"Error: 에러 메시지를 확인해주세요 {e}")
        return None, None, None

    

def check_aws():
    try:
        # Create a client for the AWS Security Token Service (STS)
        # This will use the credentials from the default profile in your credentials file
        sts_client = boto3.client('sts')
    
        # Call the get_caller_identity function to check who you are
        # This is a simple way to verify that your credentials are working
        response = sts_client.get_caller_identity()

        # Print the response which contains your Account ID, User ID, and ARN
        print("AWS credentials are set up correctly!")
        print(f"Account: {response['Account']}")
        print(f"UserId: {response['UserId']}")
        print(f"Arn: {response['Arn']}")

    except Exception as e:
        # If there is an error, it likely means credentials are not configured correctly
        print("Failed to verify AWS credentials.")
        print(f"Error: {e}")
if __name__ == '__main__':
    pipline()
    # check_aws()