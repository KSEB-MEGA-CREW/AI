# config.py

# 모델 및 경로 설정
MODEL_NAME = "ehekaanldk/kobart2ksl-translation"

#S3
S3_MODEL_PATH = "s3://mega-crew-ml-models-dev/T2G_model/V_0"
S3_GLOSSES_SET_PATH = "s3://mega-crew-ml-models-dev/point"
S3_UNI_GLOSS_SET = "s3://mega-crew-ml-models-dev/T2G_model/unique_glosses.json"
LOCAL_MODEL_PATH = './t2g_model'
LOCAL_UNI_GLOSS_SET_PATH = './preprocessed_data/unique_glosses.json'