import os
import boto3
import json
import data_load 

def map_gloss_to_point(gloss_list:list, valid_gloss_set = None):

    if valid_gloss_set:
        valid_gloss_set = data_load.load_valid_gloss_set()

    # Create an S3 client
    s3 = boto3.client('s3')

    # Name of the S3 bucket
    bucket_name = 'mega-crew-ml-models-dev'

    # This will be the final, flat list of points
    combined_point_list = []

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

    # Filter the glosses first to know the actual number of files to process
    glosses_to_process = [
        gloss for gloss in gloss_list
        if gloss in valid_gloss_set and gloss in gloss_to_file_map
        # if gloss in gloss_to_file_map
    ]

    # Iterate over the gloss list
    for i, gloss in enumerate(glosses_to_process):
        # Create the file path for the JSON file in the S3 bucket
        file_path = gloss_to_file_map.get(gloss)
        if file_path:
            try:
                print("파일 경로:"+ file_path)
                # Get the object from S3
                response = s3.get_object(Bucket=bucket_name, Key=file_path)
                # Read the content of the file
                content = response['Body'].read().decode('utf-8')
                # Parse the JSON data
                data = json.loads(content)
                
                if data:
                    # Add all frames from the current gloss
                    combined_point_list.extend(data)

                    # If it's not the last gloss, add the padding
                    if i < len(glosses_to_process) - 1:
                        last_frame = data[-1] # Get the last frame
                        combined_point_list.extend([last_frame] * 15) # Add it 15 times
            except Exception as e:
                # Print an error message if the file is not found or another error occurs
                print(f"Error fetching {file_path}: {e}")

    # Return the single combined list
    # file_path = "./temp.json"
    # with open(file_path, 'w', encoding='utf-8') as f:
    #     json.dump(combined_point_list, f, indent=4, ensure_ascii=False)

    return combined_point_list

if __name__ == '__main__':
    # 'main' 함수를 호출하여 전체 프로세스를 시작합니다.
    # map_gloss_to_point(["강2", "~대로1", "계례"])
    print(map_gloss_to_point(["사과2", "딸기", "바나나"]))