# config.py

# 모델 및 경로 설정
MODEL_NAME = "ehekaanldk/kobart2ksl-translation"

# 윈도우
# DATA_PATH = "C:/Users/Hyuk/Documents/대학/부트켐프/메인 프로젝트/AI/text_to_word/preprocessed_data/processed_data.jsonl"
# OUTPUT_DIR = "C:/Users/Hyuk/Documents/대학/부트켐프/메인 프로젝트/AI/text_to_word/kobart-finetuned-ksl-glosser"
# CUSTOM_TOKENIZER = "C:/Users/Hyuk/Documents/대학/부트켐프/메인 프로젝트/AI/text_to_word/kobart-custom-tokenizer"
# TOKENIZER_SAVE_DIR = "C:/Users/Hyuk/Documents/대학/부트켐프/메인 프로젝트/AI/text_to_word/ksl-tokenizer"

# 리눅스 1
# DATA_PATH = "/home/kdh/KSEB/0723/text_to_word/preprocessed_data/processed_data.jsonl"
# OUTPUT_DIR = "/home/kdh/KSEB/0723/text_to_word/kobart-finetuned-ksl-glosser"
# CUSTOM_TOKENIZER = "/home/kdh/KSEB/0723/text_to_word/kobart-custom-tokenizer"
# TOKENIZER_SAVE_DIR = "/home/kdh/KSEB/0723/text_to_word/ksl-tokenizer"

#리눅스 2
# DATA_PATH = "/home/202044005/KSEB/text_to_word/preprocessed_data/processed_data.jsonl"
# OUTPUT_DIR = "/home/202044005/KSEB/text_to_word/kobart-finetuned-ksl-glosser"
# CUSTOM_TOKENIZER = "/home/202044005/KSEB/text_to_word/kobart-custom-tokenizer"
# TOKENIZER_SAVE_DIR = "/home/202044005/KSEB/text_to_word/ksl-tokenizer"
# PLOTS_SAVE_DIR = "/home/202044005/KSEB/text_to_word/plots"

#S3
S3_MODEL_PATH = "s3://mega-crew-ml-models-dev/T2G_model/V_0"
S3_GLOSSES_SET_PATH = "s3://mega-crew-ml-models-dev/point"
S3_UNI_GLOSS_SET = "s3://mega-crew-ml-models-dev/T2G_model/unique_glosses.json"
LOCAL_MODEL_PATH = './t2g_model'
LOCAL_UNI_GLOSS_SET_PATH = './preprocessed_data/unique_glosses.json'