import os
import boto3
import json
import util.data_load as data_load

def validate_gloss(gloss:str, valid_glosses:set):
    """
    Validate if the gloss is in the set of valid glosses.
    """
    if gloss not in valid_glosses:
        # print(f"Warning: '{gloss}' is not a valid gloss.")
        return False
    return True

def map_gloss_to_point(gloss_list:list):

    valid_gloss_set = data_load.load_valid_gloss_set()

    # Create an S3 client
    s3 = boto3.client('s3')

    # Name of the S3 bucket
    bucket_name = 'mega-crew-ml-models-dev'

    # List to store the JSON data from S3
    point_list = []

    s3_file_keys = []
    try:
        #paginator를 사용하여 버킷을 많은 오브젝트로 구성할 수 있습니다.
        paginator = s3.get_paginator('list_objects_v2')
        # Create a paginator for the list_objects_v2 operation.
        # list_objects_v2 작업을 하기 위해 pagenator를 생성합니다.
        #list_objects_v2 is used to list the objects in an S3 bucket.
        # list_objects_v2는 S3 버킷의 객채를 나열하기 위해 사용된다.
        # Paginate through the objects in the specified bucket and prefix.
        # 지정된 버킷과 접두사에 있는 객체를 페이지로 나누어 가져옵니다.
        pages = paginator.paginate(Bucket=bucket_name, Prefix='point/')
        # Iterate through each page of results.
        for page in pages:
            # Check if the page contains a list of objects.    
            if 'Contents' in page:
                # Iterate through each object in the contents.
                for obj in page['Contents']:
                    # Append the object key to the list.
                    s3_file_keys.append(obj['Key'])
    except Exception as e:
        print(f"Error listing files from S3: {e}")
        return []
    gloss_to_file_map = {}
    for key in s3_file_keys:
        if not key.endswith('json'):
            continue
        filename = os.path.basename(key)
        gloss_candidate = os.path.splitext(filename)[0]
        parts = gloss_candidate.split('_')
        actual_gloss = parts[-1]
        gloss_to_file_map[actual_gloss] = key











































































    # Iterate over the gloss list
    for gloss in gloss_list:
        if gloss not in valid_gloss_set:
            continue
        # Create the file path for the JSON file in the S3 bucket
        file_path = gloss_to_file_map.get(gloss)
        if file_path:
            try:
                # Get the object from S3
                response = s3.get_object(Bucket=bucket_name, Key=file_path)
                # Read the content of the file
                content = response['Body'].read().decode('utf-8')
                # Parse the JSON data
                data = json.loads(content)
                # Append the data to the list
                point_list.append(data)
            except Exception as e:
                # Print an error message if the file is not found or another error occurs
                print(f"Error fetching {file_path}: {e}")

    # Return the list of JSON data
    return point_list

if __name__ == '__main__':
    # 'main' 함수를 호출하여 전체 프로세스를 시작합니다.
    print(map_gloss_to_point(["사과", "dd", "경제3"]))