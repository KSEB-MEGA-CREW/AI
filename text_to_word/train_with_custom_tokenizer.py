# 'json' 라이브러리를 임포트합니다. JSON 형식의 데이터를 다루기 위해 필요합니다. (예: 파일 읽기/쓰기)
import json
# 'torch' 라이브러리를 임포트합니다. 파이토치는 딥러닝 모델을 구축하고 학습시키는 데 사용되는 핵심 프레임워크입니다.
import torch
# 'os' 라이브러리를 임포트합니다. 파일 시스템 경로를 다루기 위해 필요합니다.
import os
# 'datasets' 라이브러리에서 'Dataset' 클래스를 임포트합니다. Hugging Face의 데이터셋을 쉽게 다루기 위해 사용됩니다.
from datasets import Dataset
# 'transformers' 라이브러리에서 필요한 클래스들을 임포트합니다.
from transformers import (
    # 'AutoTokenizer'는 사전 학습된 모델에 맞는 토크나이저를 자동으로 로드하는 클래스입니다.
    AutoTokenizer,
    # 'AutoModelForSeq2SeqLM'은 텍스트-투-텍스트(번역, 요약 등) 과업을 위한 사전 학습된 모델을 자동으로 로드하는 클래스입니다.
    AutoModelForSeq2SeqLM,
    # 'Seq2SeqTrainingArguments'는 Seq2Seq 모델의 학습 과정을 설정하는 다양한 인자(파라미터)를 정의하는 클래스입니다.
    Seq2SeqTrainingArguments,
    # 'Seq2SeqTrainer'는 Seq2Seq 모델의 학습 및 평가를 쉽게 할 수 있도록 돕는 고수준 API 클래스입니다.
    Seq2SeqTrainer,
    # 'DataCollatorForSeq2Seq'는 Seq2Seq 모델의 학습을 위해 배치(batch) 단위로 데이터를 동적으로 패딩(padding) 처리하는 클래스입니다.
    DataCollatorForSeq2Seq
)
# 'data_load' 모듈을 임포트합니다. 데이터 로딩을 위한 사용자 정의 함수가 포함되어 있습니다.
import util.data_load as data_load
import config

# 스크립트의 메인 로직을 포함하는 'main' 함수를 정의합니다.
def main():
    # 1. 기본 변수 설정
    # 사용할 사전 학습된 KoBART 모델의 이름을 Hugging Face 허브에서 가져와 설정합니다.
    MODEL_NAME = config.MODEL_NAME
    # 학습에 사용할 데이터 파일(.jsonl)의 경로를 설정합니다.
    DATA_PATH = config.DATA_PATH
    # 데이터셋의 모든 고유 gloss를 추가하여 만든 커스텀 토크나이저를 저장할 디렉토리 경로를 설정합니다.
    TOKENIZER_SAVE_DIR = config.NEW_MODEL_DIR
    # 파인튜닝된 모델과 최종 토크나이저, 그리고 학습 로그가 저장될 디렉토리 경로를 설정합니다.
    OUTPUT_DIR = config.OUTPUT_DIR

    # NEW_MODEL_DIR = config.NEW_MODEL_DIR

    # 위에서 정의한 'load_jsonl' 함수를 호출하여 학습 데이터를 로드합니다.
    train_data = data_load.load_jsonl(DATA_PATH, limit=1000)

    # 2. 커스텀 토크나이저 생성 및 저장 (존재하지 않을 경우)
    # 만약 커스텀 토크나이저가 저장된 디렉토리가 없다면 새로 생성합니다.
    if not os.path.exists(TOKENIZER_SAVE_DIR):
        # 'print' 함수로 커스텀 토크나이저 생성을 시작함을 알립니다.
        print(f"'{TOKENIZER_SAVE_DIR}' 경로에 커스텀 토크나이저가 없습니다. 새로 생성합니다.")
        
        # 기본 KoBART 토크나이저를 로드합니다.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 데이터셋에 있는 모든 고유한 gloss를 추출하기 위한 빈 집합(set)을 생성합니다.
        unique_glosses = set()
        # 학습 데이터 전체를 순회합니다.
        for item in train_data:
            # 각 데이터의 'gloss_id' 리스트에 있는 모든 gloss에 대해 반복합니다.
            for gloss in item['gloss_id']:
                # 집합에 gloss를 추가하여 중복을 자동으로 제거합니다.
                unique_glosses.add(gloss)
        
        # 집합을 리스트로 변환합니다.
        new_tokens = list(unique_glosses)
        
        # 'add_tokens' 메소드를 사용하여 기존 토크나이저의 어휘 사전에 새로운 gloss 토큰들을 추가합니다.
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        
        # 'print' 함수와 f-string으로 몇 개의 새로운 토큰이 추가되었는지 출력합니다.
        print(f"{num_added_tokens}개의 새로운 gloss 토큰을 어휘 사전에 추가했습니다.")
        
        # 'os.makedirs'를 사용하여 커스텀 토크나이저를 저장할 디렉토리를 생성합니다.
        os.makedirs(TOKENIZER_SAVE_DIR)
        # 'save_pretrained' 메소드를 사용하여 새로운 토큰이 추가된 토크나이저를 지정된 경로에 저장합니다.
        tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)
        # 'print' 함수와 f-string으로 토크나이저가 어디에 저장되었는지 알려줍니다.
        print(f"어휘 사전이 확장된 토크나이저를 '{TOKENIZER_SAVE_DIR}'에 저장했습니다.")

    # 3. 커스텀 토크나이저 로드
    # 'print' 함수와 f-string으로 저장된 커스텀 토크나이저를 불러온다는 것을 알립니다.
    print(f"'{TOKENIZER_SAVE_DIR}'에서 커스텀 토크나이저를 로드합니다.")
    # 'AutoTokenizer.from_pretrained'를 사용하여 어휘 사전이 확장된 커스텀 토크나이저를 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SAVE_DIR)

    # 4. 데이터셋 준비
    # Hugging Face의 'Dataset' 객체로 변환하기 위해 데이터를 딕셔너리 형태로 가공합니다.
    processed_data = {
        # 'koreanText' 키에 모든 한국어 문장 리스트를 할당합니다.
        "koreanText": [item['koreanText'] for item in train_data],
        # 'gloss_id' 키에 공백으로 합쳐진 수어 단어(gloss) 문자열 리스트를 할당합니다.
        "gloss_id": [" ".join(item['gloss_id']) for item in train_data]
    }
    # 가공된 파이썬 딕셔너리를 Hugging Face의 'Dataset' 객체로 변환합니다.
    dataset = Dataset.from_dict(processed_data)

    # 전체 데이터셋을 훈련용(90%)과 검증용(10%)으로 분리합니다.
    dataset = dataset.train_test_split(test_size=0.1)
    # 훈련용 데이터셋을 'train_dataset' 변수에 할당합니다.
    train_dataset = dataset['train']
    # 검증용 데이터셋을 'eval_dataset' 변수에 할당합니다.
    eval_dataset = dataset['test']

    # 5. 데이터 토큰화
    # 입력 시퀀스(한국어 문장)의 최대 길이를 128 토큰으로 설정합니다.
    max_input_length = 128
    # 타겟 시퀀스(수어 단어)의 최대 길이를 128 토큰으로 설정합니다.
    max_target_length = 128

    # 데이터셋을 모델 입력 형식에 맞게 토큰화하는 함수를 정의합니다.
    def preprocess_function(examples):
        # 입력으로 사용할 'koreanText' 컬럼의 텍스트 리스트를 'inputs'에 저장합니다.
        inputs = [ex for ex in examples['koreanText']]
        # 정답으로 사용할 'gloss_id' 컬럼의 텍스트 리스트를 'targets'에 저장합니다.
        targets = [ex for ex in examples['gloss_id']]
        # 입력 텍스트를 토큰화합니다.
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
        # 타겟(레이블) 텍스트를 토큰화합니다.
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
        # 모델의 정답으로 사용될 'labels'를 'model_inputs'에 추가합니다.
        model_inputs["labels"] = labels["input_ids"]
        # 전처리된 결과를 반환합니다.
        return model_inputs

    # 'map' 함수를 사용하여 훈련 및 검증 데이터셋 전체에 전처리 함수를 일괄 적용합니다.
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # 6. 모델 로드 및 임베딩 크기 조정
    # 'AutoModelForSeq2SeqLM.from_pretrained'를 사용해 사전 학습된 모델을 로드합니다.
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # 'resize_token_embeddings' 메소드를 호출하여 모델의 토큰 임베딩 레이어 크기를 새로운 토크나이저의 어휘 사전 크기에 맞게 조정합니다.
    # 이는 새로운 토큰에 대한 임베딩 벡터를 생성하기 위해 필수적인 과정입니다.
    model.resize_token_embeddings(len(tokenizer))
    # 'print' 함수와 f-string으로 모델의 임베딩 크기가 조정되었음을 알립니다.
    print(f"모델의 토큰 임베딩 크기를 {len(tokenizer)}에 맞게 조정했습니다.")

    # 7. 학습 인자(Arguments) 설정 (문제 해결을 위해 최소한의 인자만 사용)
    # 'Seq2SeqTrainingArguments' 클래스를 사용하여 모델 학습 설정을 정의합니다.
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR, # 모델과 결과물이 저장될 경로
        num_train_epochs=3,    # 전체 훈련 에포크 수
        per_device_train_batch_size=8, # 훈련용 배치 크기
        logging_steps=100,     # 로그 출력 주기
        # predict_with_generate=True, # 추후 필요시 활성화
        # fp16=torch.cuda.is_available(), # 추후 필요시 활성화
    )

    # 8. 데이터 콜레이터 정의
    # 'DataCollatorForSeq2Seq' 객체를 생성하여 배치의 동적 패딩을 처리합니다.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    # 9. 트레이너(Trainer) 정의 및 학습 시작 (평가 없이)
    # 'Seq2SeqTrainer' 객체를 생성하여 학습 과정을 총괄합니다.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        # eval_dataset=tokenized_eval_dataset, # 평가를 비활성화하므로 주석 처리
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 'print' 함수로 모델 파인튜닝 시작을 알립니다.
    print("모델 파인튜닝을 시작합니다.")
    # 'trainer.train()' 메소드를 호출하여 실제 모델 학습을 시작합니다.
    trainer.train()
    # 'print' 함수로 모델 파인튜닝이 완료되었음을 알립니다.
    print("모델 파인튜닝이 완료되었습니다.")

    # 10. 학습된 모델과 토크나이저 저장
    # 'print' 함수와 f-string을 사용하여 학습된 모델이 어디에 저장되는지 알려줍니다.
    print(f"학습된 모델을 '{OUTPUT_DIR}'에 저장합니다.")
    # 'trainer.save_model()'을 호출하여 최종 모델을 저장합니다.
    trainer.save_model(OUTPUT_DIR)
    # 'tokenizer.save_pretrained()'를 호출하여 최종 토크나이저를 저장합니다.
    tokenizer.save_pretrained(OUTPUT_DIR)

# 이 스크립트 파일이 직접 실행될 때만 'main()' 함수를 호출하도록 합니다.
if __name__ == '__main__':
    # 'main' 함수를 호출하여 전체 프로세스를 시작합니다.
    main()
