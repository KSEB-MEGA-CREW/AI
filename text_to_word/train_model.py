# 'json' 라이브러리를 임포트합니다. JSON 형식의 데이터를 다루기 위해 필요합니다. (예: 파일 읽기/쓰기)
import json
# 'torch' 라이브러리를 임포트합니다. 파이토치는 딥러닝 모델을 구축하고 학습시키는 데 사용되는 핵심 프레임워크입니다.
import torch
# 'datasets' 라이브러리에서 'load_dataset'와 'Dataset' 클래스를 임포트합니다. Hugging Face의 데이터셋을 쉽게 다루기 위해 사용됩니다.
from datasets import load_dataset, Dataset
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
import util.data_load as data_load
import config
import util.t2g_tokenizer as t2g_tokenizer
# 1. 기본 변수 설정
# 사용할 사전 학습된 KoBART 모델의 이름을 Hugging Face 허브에서 가져와 설정합니다.
MODEL_NAME = config.MODEL_NAME
# 학습에 사용할 데이터 파일(.jsonl)의 경로를 설정합니다.
DATA_PATH = config.DATA_PATH
# 파인튜닝된 모델과 토크나이저, 그리고 학습 로그가 저장될 디렉토리 경로를 설정합니다.
OUTPUT_DIR = config.OUTPUT_DIR

# 입력 시퀀스(한국어 문장)의 최대 길이를 128 토큰으로 설정합니다. 이보다 길면 잘라냅니다.
max_input_length = 128
# 타겟 시퀀스(수어 단어)의 최대 길이를 128 토큰으로 설정합니다.
max_target_length = 128


def data_train():
    
    # 위에서 정의한 'load_jsonl' 함수를 호출하여 학습 데이터를 로드합니다.
    train_data = data_load.load_jsonl(DATA_PATH)
    
    # Hugging Face의 'Dataset' 객체로 변환하기 위해 데이터를 딕셔너리 형태로 가공합니다.
    # 'gloss_id'는 원래 리스트 형태이므로, 각 요소를 공백으로 연결하여 하나의 문자열로 변환합니다.
    processed_data = {
        # 'koreanText' 키에 모든 한국어 문장 리스트를 할당합니다.
        "koreanText": [item['koreanText'] for item in train_data],
        # 'gloss_id' 키에 공백으로 합쳐진 수어 단어(gloss) 문자열 리스트를 할당합니다.
        "gloss_id": [" ".join(item['gloss_id']) for item in train_data]
    }
    # print(processed_data['gloss_id'])

    #############################################################
    # 2. Data Preparation

    # 가공된 파이썬 딕셔너리를 Hugging Face의 'Dataset' 객체로 변환합니다.
    dataset = Dataset.from_dict(processed_data)

    # 전체 데이터셋을 훈련용(90%)과 검증용(10%)으로 분리합니다. 'test_size=0.1'은 10%를 검증용으로 사용하겠다는 의미입니다.
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # 분리된 데이터셋에서 훈련용 데이터셋을 'train_dataset' 변수에 할당합니다.
    train_dataset = dataset['train']
    # 분리된 데이터셋에서 검증용 데이터셋을 'eval_dataset' 변수에 할당합니다.
    eval_dataset = dataset['test']

    # 'print' 함수와 f-string을 사용하여 훈련 데이터셋의 크기를 출력합니다.
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    # 'print' 함수와 f-string을 사용하여 검증 데이터셋의 크기를 출력합니다.
    print(f"검증 데이터셋 크기: {len(eval_dataset)}")
    # 데이터가 잘 로드되었는지 확인하기 위해 훈련 데이터셋의 첫 번째 샘플을 출력합니다.
    print("데이터셋 샘플:")
    # 'train_dataset[0]'은 훈련 데이터셋의 첫 번째 데이터를 의미합니다.
    # print(train_dataset[0])

    # 3. 모델, 토크나이저 로드
    # 'AutoTokenizer.from_pretrained'를 사용해 'MODEL_NAME'에 해당하는 사전 학습된 모델의 토크나이저를 로드합니다.
    model, tokenizer = t2g_tokenizer.load_model()
    # # 입력 시퀀스(한국어 문장)의 최대 길이를 128 토큰으로 설정합니다. 이보다 길면 잘라냅니다.
    # max_input_length = 128
    # # 타겟 시퀀스(수어 단어)의 최대 길이를 128 토큰으로 설정합니다.
    # max_target_length = 128

    # 데이터셋을 모델 입력 형식에 맞게 토큰화하고 변환하는 함수를 정의합니다.
    def preprocess_function(examples):
        # 입력 데이터로 사용할 'koreanText' 컬럼의 텍스트 리스트를 'inputs'에 저장합니다.
        inputs = [ex for ex in examples['koreanText']]
        # 정답(레이블) 데이터로 사용할 'gloss_id' 컬럼의 텍스트 리스트를 'targets'에 저장합니다.
        targets = [ex for ex in examples['gloss_id']]
        
        # 입력 텍스트('inputs')를 토큰화합니다. 'max_length'로 최대 길이를, 'truncation=True'로 길이 초과 시 자르기를, 'padding="max_length"'로 최대 길이에 맞춰 패딩을 설정합니다.
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

        # 'with tokenizer.as_target_tokenizer()' 컨텍스트 매니저를 사용하여 타겟(레이블) 텍스트를 토큰화합니다. 이는 Seq2Seq 모델에서 디코더 입력을 올바르게 처리하기 위함입니다.
        with tokenizer.as_target_tokenizer():
            # 타겟 텍스트('targets')를 토큰화합니다. 입력과 동일하게 최대 길이 설정, 자르기, 패딩을 적용합니다.
            labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

        # 모델이 정답으로 사용할 수 있도록, 토큰화된 레이블의 'input_ids'를 'model_inputs' 딕셔너리의 "labels" 키에 추가합니다.
        model_inputs["labels"] = labels["input_ids"]
        # 전처리된 'model_inputs' 딕셔너리를 반환합니다.
        return model_inputs

    # Apply the preprocess_function to the entire training dataset using the 'map' function.
    # 'batched=True' processes multiple samples at once to speed up the process.
    # Use 'remove_columns' to delete the original text columns, keeping only the columns needed by the model.
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    # print(f"tokenized_train_dataset type : {type(tokenized_train_dataset)}")
    # Apply the same preprocessing function to the entire validation dataset.
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # The KoBART model does not use token_type_ids, so if this column exists, remove it.
    # Without this step, the Trainer might pass an unnecessary argument to the model, causing an error.
    if 'token_type_ids' in tokenized_train_dataset.column_names:
        # Remove the 'token_type_ids' column from the training dataset.
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(['token_type_ids'])
        # Remove the 'token_type_ids' column from the validation dataset.
        tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(['token_type_ids'])

    # 6. 학습 인자(Arguments) 설정
    # 'Seq2SeqTrainingArguments' 클래스를 사용하여 모델 학습에 필요한 다양한 하이퍼파라미터와 설정을 정의합니다.
    training_args = Seq2SeqTrainingArguments(
        # 학습 결과물(체크포인트, 모델 등)이 저장될 디렉토리를 지정합니다.
        output_dir=OUTPUT_DIR,
        # 평가(evaluation)를 언제 수행할지 결정합니다. "epoch"은 한 에포크가 끝날 때마다 평가를 수행하라는 의미입니다.
        eval_strategy="steps",
        # 학습률(learning rate)을 2e-5 (0.00002)로 설정합니다.
        learning_rate=2e-5,
        # 훈련 시 각 GPU(device) 당 배치 크기를 8로 설정합니다. GPU 메모리에 따라 조절해야 합니다.
        per_device_train_batch_size=16,
        # 평가 시 각 GPU(device) 당 배치 크기를 8로 설정합니다.
        per_device_eval_batch_size=16,
        # 가중치 감소(weight decay) 정규화 값을 0.01로 설정하여 과적합을 방지합니다.
        weight_decay=0.01,
        # 저장할 최대 체크포인트 수를 3으로 제한합니다. 이 수를 넘으면 가장 오래된 체크포인트가 삭제됩니다.
        save_total_limit=3,
        # 전체 훈련 데이터셋을 총 3번 반복하여 학습(에포크 수)하도록 설정합니다.
        num_train_epochs=5,
        # 평가 시 'model.generate()'를 사용하여 BLEU, ROUGE 같은 생성 기반 메트릭을 계산할 수 있도록 설정합니다.
        predict_with_generate=True,
        # 'torch.cuda.is_available()'로 CUDA 지원 GPU가 있을 경우, fp16(16비트 부동소수점) 혼합 정밀도 학습을 활성화하여 메모리 사용량을 줄이고 학습 속도를 높입니다.
        fp16=torch.cuda.is_available(),
        # 학습 로그가 저장될 디렉토리를 지정합니다.
        logging_dir=f'{OUTPUT_DIR}/logs',
        # 100 스텝(step)마다 학습 로그(loss 등)를 기록하도록 설정합니다.
        logging_steps=100,
        # 모델 체크포인트를 저장하는 주기를 "epoch"으로 설정하여, 매 에포크마다 저장합니다.
        save_strategy="steps",
        # 훈련이 끝났을 때 가장 성능이 좋았던 모델을 자동으로 로드하도록 설정합니다.
        load_best_model_at_end=True,
        # '최고의 모델'을 결정하는 기준이 될 메트릭을 'eval_loss'(검증 손실)로 지정합니다.
        metric_for_best_model="eval_loss",
        # 위에서 설정한 메트릭('eval_loss')의 값이 작을수록 더 좋은 모델임을 명시합니다. (loss는 낮을수록 좋음)
        greater_is_better=False,
        # Set a seed for reproducible training. This ensures that aspects like model
        # weight initialization and data shuffling are the same for each run.
        seed=42,
    )

    # 7. 데이터 콜레이터 정의
    # 'DataCollatorForSeq2Seq' 객체를 생성합니다. 이 객체는 배치 내의 시퀀스들을 동적으로 패딩하여 길이를 맞추는 역할을 합니다.
    data_collator = DataCollatorForSeq2Seq(
        # 사용할 토크나이저를 지정합니다.
        tokenizer,
        # 사용할 모델을 지정합니다. (모델 아키텍처에 따라 패딩 방식이 달라질 수 있음)
        model=model
    )

    # 8. 트레이너(Trainer) 정의 및 학습 시작
    # 'Seq2SeqTrainer' 객체를 생성하여 학습 과정을 총괄하도록 합니다.
    trainer = Seq2SeqTrainer(
        # 파인튜닝할 모델을 전달합니다.
        model=model,
        # 위에서 정의한 학습 관련 인자들을 전달합니다.
        args=training_args,
        # 토큰화된 훈련 데이터셋을 전달합니다.
        train_dataset=tokenized_train_dataset,
        # 토큰화된 검증 데이터셋을 전달합니다.
        eval_dataset=tokenized_eval_dataset,
        # 토크나이저를 전달합니다. (생성된 텍스트를 디코딩하는 데 사용됨)
        tokenizer=tokenizer,
        # 데이터 콜레이터를 전달하여 배치를 구성합니다.
        data_collator=data_collator,
    )

    # 'print' 함수로 모델 파인튜닝 시작을 알립니다.
    print("모델 파인튜닝을 시작합니다.")
    # 'trainer.train()' 메소드를 호출하여 실제 모델 학습을 시작합니다.
    trainer.train()
    # 'print' 함수로 모델 파인튜닝이 완료되었음을 알립니다.
    print("모델 파인튜닝이 완료되었습니다.")

    # 9. 학습된 모델과 토크나이저 저장
    # 'print' 함수와 f-string을 사용하여 학습된 모델이 어디에 저장되는지 알려줍니다.
    print(f"학습된 모델을 '{OUTPUT_DIR}'에 저장합니다.")
    # 'trainer.save_model()'을 호출하여 최종적으로 가장 성능이 좋았던 모델의 가중치와 설정을 지정된 경로에 저장합니다.
    trainer.save_model(OUTPUT_DIR)
    # 'tokenizer.save_pretrained()'를 호출하여 파인튜닝에 사용된 토크나이저의 정보(vocabs 등)를 같은 경로에 저장합니다.
    tokenizer.save_pretrained(OUTPUT_DIR)



# 스크립트의 메인 로직을 포함하는 'main' 함수를 정의합니다.
def main():
    '''
    hh
    '''
    '''
    # 1. 기본 변수 설정
    # 사용할 사전 학습된 KoBART 모델의 이름을 Hugging Face 허브에서 가져와 설정합니다.
    MODEL_NAME = config.MODEL_NAME
    # 학습에 사용할 데이터 파일(.jsonl)의 경로를 설정합니다.
    DATA_PATH = config.DATA_PATH
    # 파인튜닝된 모델과 토크나이저, 그리고 학습 로그가 저장될 디렉토리 경로를 설정합니다.
    OUTPUT_DIR = config.OUTPUT_DIR

    # 위에서 정의한 'load_jsonl' 함수를 호출하여 학습 데이터를 로드합니다.
    train_data = data_load.load_jsonl(DATA_PATH, limit = 1000)
    
    # Hugging Face의 'Dataset' 객체로 변환하기 위해 데이터를 딕셔너리 형태로 가공합니다.
    # 'gloss_id'는 원래 리스트 형태이므로, 각 요소를 공백으로 연결하여 하나의 문자열로 변환합니다.
    processed_data = {
        # 'koreanText' 키에 모든 한국어 문장 리스트를 할당합니다.
        "koreanText": [item['koreanText'] for item in train_data],
        # 'gloss_id' 키에 공백으로 합쳐진 수어 단어(gloss) 문자열 리스트를 할당합니다.
        "gloss_id": [" ".join(item['gloss_id']) for item in train_data]
    }
    # print(processed_data['gloss_id'])

    #############################################################
    # 2. Data Preparation

    # 가공된 파이썬 딕셔너리를 Hugging Face의 'Dataset' 객체로 변환합니다.
    dataset = Dataset.from_dict(processed_data)

    # 전체 데이터셋을 훈련용(90%)과 검증용(10%)으로 분리합니다. 'test_size=0.1'은 10%를 검증용으로 사용하겠다는 의미입니다.
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    # 분리된 데이터셋에서 훈련용 데이터셋을 'train_dataset' 변수에 할당합니다.
    train_dataset = dataset['train']
    # 분리된 데이터셋에서 검증용 데이터셋을 'eval_dataset' 변수에 할당합니다.
    eval_dataset = dataset['test']

    # 'print' 함수와 f-string을 사용하여 훈련 데이터셋의 크기를 출력합니다.
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    # 'print' 함수와 f-string을 사용하여 검증 데이터셋의 크기를 출력합니다.
    print(f"검증 데이터셋 크기: {len(eval_dataset)}")
    # 데이터가 잘 로드되었는지 확인하기 위해 훈련 데이터셋의 첫 번째 샘플을 출력합니다.
    print("데이터셋 샘플:")
    # 'train_dataset[0]'은 훈련 데이터셋의 첫 번째 데이터를 의미합니다.
    # print(train_dataset[0])

    # 3. 모델, 토크나이저 로드
    # 'AutoTokenizer.from_pretrained'를 사용해 'MODEL_NAME'에 해당하는 사전 학습된 모델의 토크나이저를 로드합니다.
    model, tokenizer = t2g_tokenizer.load_model()
    # 입력 시퀀스(한국어 문장)의 최대 길이를 128 토큰으로 설정합니다. 이보다 길면 잘라냅니다.
    max_input_length = 128
    # 타겟 시퀀스(수어 단어)의 최대 길이를 128 토큰으로 설정합니다.
    max_target_length = 128

    # 데이터셋을 모델 입력 형식에 맞게 토큰화하고 변환하는 함수를 정의합니다.
    def preprocess_function(examples):
        # 입력 데이터로 사용할 'koreanText' 컬럼의 텍스트 리스트를 'inputs'에 저장합니다.
        inputs = [ex for ex in examples['koreanText']]
        # 정답(레이블) 데이터로 사용할 'gloss_id' 컬럼의 텍스트 리스트를 'targets'에 저장합니다.
        targets = [ex for ex in examples['gloss_id']]
        
        # 입력 텍스트('inputs')를 토큰화합니다. 'max_length'로 최대 길이를, 'truncation=True'로 길이 초과 시 자르기를, 'padding="max_length"'로 최대 길이에 맞춰 패딩을 설정합니다.
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

        # 'with tokenizer.as_target_tokenizer()' 컨텍스트 매니저를 사용하여 타겟(레이블) 텍스트를 토큰화합니다. 이는 Seq2Seq 모델에서 디코더 입력을 올바르게 처리하기 위함입니다.
        with tokenizer.as_target_tokenizer():
            # 타겟 텍스트('targets')를 토큰화합니다. 입력과 동일하게 최대 길이 설정, 자르기, 패딩을 적용합니다.
            labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

        # 모델이 정답으로 사용할 수 있도록, 토큰화된 레이블의 'input_ids'를 'model_inputs' 딕셔너리의 "labels" 키에 추가합니다.
        model_inputs["labels"] = labels["input_ids"]
        # 전처리된 'model_inputs' 딕셔너리를 반환합니다.
        return model_inputs

    # Apply the preprocess_function to the entire training dataset using the 'map' function.
    # 'batched=True' processes multiple samples at once to speed up the process.
    # Use 'remove_columns' to delete the original text columns, keeping only the columns needed by the model.
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    print(f"tokenized_train_dataset type : {type(tokenized_train_dataset)}")
    # Apply the same preprocessing function to the entire validation dataset.
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # The KoBART model does not use token_type_ids, so if this column exists, remove it.
    # Without this step, the Trainer might pass an unnecessary argument to the model, causing an error.
    if 'token_type_ids' in tokenized_train_dataset.column_names:
        # Remove the 'token_type_ids' column from the training dataset.
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(['token_type_ids'])
        # Remove the 'token_type_ids' column from the validation dataset.
        tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(['token_type_ids'])

    # 6. 학습 인자(Arguments) 설정
    # 'Seq2SeqTrainingArguments' 클래스를 사용하여 모델 학습에 필요한 다양한 하이퍼파라미터와 설정을 정의합니다.
    training_args = Seq2SeqTrainingArguments(
        # 학습 결과물(체크포인트, 모델 등)이 저장될 디렉토리를 지정합니다.
        output_dir=OUTPUT_DIR,
        # 평가(evaluation)를 언제 수행할지 결정합니다. "epoch"은 한 에포크가 끝날 때마다 평가를 수행하라는 의미입니다.
        eval_strategy="steps",
        # 학습률(learning rate)을 2e-5 (0.00002)로 설정합니다.
        learning_rate=2e-5,
        # 훈련 시 각 GPU(device) 당 배치 크기를 8로 설정합니다. GPU 메모리에 따라 조절해야 합니다.
        per_device_train_batch_size=8,
        # 평가 시 각 GPU(device) 당 배치 크기를 8로 설정합니다.
        per_device_eval_batch_size=8,
        # 가중치 감소(weight decay) 정규화 값을 0.01로 설정하여 과적합을 방지합니다.
        weight_decay=0.01,
        # 저장할 최대 체크포인트 수를 3으로 제한합니다. 이 수를 넘으면 가장 오래된 체크포인트가 삭제됩니다.
        save_total_limit=3,
        # 전체 훈련 데이터셋을 총 3번 반복하여 학습(에포크 수)하도록 설정합니다.
        num_train_epochs=1,
        # 평가 시 'model.generate()'를 사용하여 BLEU, ROUGE 같은 생성 기반 메트릭을 계산할 수 있도록 설정합니다.
        predict_with_generate=True,
        # 'torch.cuda.is_available()'로 CUDA 지원 GPU가 있을 경우, fp16(16비트 부동소수점) 혼합 정밀도 학습을 활성화하여 메모리 사용량을 줄이고 학습 속도를 높입니다.
        fp16=torch.cuda.is_available(),
        # 학습 로그가 저장될 디렉토리를 지정합니다.
        logging_dir=f'{OUTPUT_DIR}/logs',
        # 100 스텝(step)마다 학습 로그(loss 등)를 기록하도록 설정합니다.
        logging_steps=100,
        # 모델 체크포인트를 저장하는 주기를 "epoch"으로 설정하여, 매 에포크마다 저장합니다.
        save_strategy="steps",
        # 훈련이 끝났을 때 가장 성능이 좋았던 모델을 자동으로 로드하도록 설정합니다.
        load_best_model_at_end=True,
        # '최고의 모델'을 결정하는 기준이 될 메트릭을 'eval_loss'(검증 손실)로 지정합니다.
        metric_for_best_model="eval_loss",
        # 위에서 설정한 메트릭('eval_loss')의 값이 작을수록 더 좋은 모델임을 명시합니다. (loss는 낮을수록 좋음)
        greater_is_better=False,
        # Set a seed for reproducible training. This ensures that aspects like model
        # weight initialization and data shuffling are the same for each run.
        seed=42,
    )

    # 7. 데이터 콜레이터 정의
    # 'DataCollatorForSeq2Seq' 객체를 생성합니다. 이 객체는 배치 내의 시퀀스들을 동적으로 패딩하여 길이를 맞추는 역할을 합니다.
    data_collator = DataCollatorForSeq2Seq(
        # 사용할 토크나이저를 지정합니다.
        tokenizer,
        # 사용할 모델을 지정합니다. (모델 아키텍처에 따라 패딩 방식이 달라질 수 있음)
        model=model
    )

    # 8. 트레이너(Trainer) 정의 및 학습 시작
    # 'Seq2SeqTrainer' 객체를 생성하여 학습 과정을 총괄하도록 합니다.
    trainer = Seq2SeqTrainer(
        # 파인튜닝할 모델을 전달합니다.
        model=model,
        # 위에서 정의한 학습 관련 인자들을 전달합니다.
        args=training_args,
        # 토큰화된 훈련 데이터셋을 전달합니다.
        train_dataset=tokenized_train_dataset,
        # 토큰화된 검증 데이터셋을 전달합니다.
        eval_dataset=tokenized_eval_dataset,
        # 토크나이저를 전달합니다. (생성된 텍스트를 디코딩하는 데 사용됨)
        tokenizer=tokenizer,
        # 데이터 콜레이터를 전달하여 배치를 구성합니다.
        data_collator=data_collator,
    )

    # 'print' 함수로 모델 파인튜닝 시작을 알립니다.
    print("모델 파인튜닝을 시작합니다.")
    # 'trainer.train()' 메소드를 호출하여 실제 모델 학습을 시작합니다.
    trainer.train()
    # 'print' 함수로 모델 파인튜닝이 완료되었음을 알립니다.
    print("모델 파인튜닝이 완료되었습니다.")

    # 9. 학습된 모델과 토크나이저 저장
    # 'print' 함수와 f-string을 사용하여 학습된 모델이 어디에 저장되는지 알려줍니다.
    print(f"학습된 모델을 '{OUTPUT_DIR}'에 저장합니다.")
    # 'trainer.save_model()'을 호출하여 최종적으로 가장 성능이 좋았던 모델의 가중치와 설정을 지정된 경로에 저장합니다.
    trainer.save_model(OUTPUT_DIR)
    # 'tokenizer.save_pretrained()'를 호출하여 파인튜닝에 사용된 토크나이저의 정보(vocabs 등)를 같은 경로에 저장합니다.
    tokenizer.save_pretrained(OUTPUT_DIR)
    '''
    # 10. 추론(Inference) 예시
    # 'print' 함수로 추론 테스트 섹션의 시작을 알립니다.
    print("\n--- 추론 테스트 ---")
    # 'AutoModelForSeq2SeqLM.from_pretrained'를 사용하여 방금 저장한 파인튜닝된 모델을 다시 불러옵니다.
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
    # 'AutoTokenizer.from_pretrained'를 사용하여 저장된 토크나이저를 다시 불러옵니다.
    trained_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    # 번역(gloss 생성)을 테스트할 한국어 문장을 정의합니다.
    text_to_translate = "안녕하세요 만나서 반갑습니다"
    # 'print' 함수와 f-string으로 어떤 문장이 입력되었는지 출력합니다.
    print(f"입력 문장: {text_to_translate}")

    # 입력 문장을 토큰화합니다. 'return_tensors="pt"'는 결과를 파이토치 텐서로 반환하라는 의미입니다. '.to(trained_model.device)'는 모델이 있는 장치(CPU 또는 GPU)로 텐서를 이동시킵니다.
    inputs = trained_tokenizer(text_to_translate, return_tensors="pt", max_length=max_input_length, truncation=True).to(trained_model.device)
    # print(f"inputs type : {type(inputs)}")
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
        print(f"input에서 token_type_ids가 제거되었습니다.")

    # 'trained_model.generate()' 메소드를 호출하여 입력 토큰(**inputs)으로부터 새로운 텍스트(gloss)를 생성합니다.
    outputs = trained_model.generate(**inputs, max_length=max_target_length)
    
    # 'trained_tokenizer.decode()'를 사용하여 모델이 생성한 토큰 ID 시퀀스('outputs[0]')를 사람이 읽을 수 있는 문자열로 변환합니다. 'skip_special_tokens=True'는 <pad>, <eos> 같은 특수 토큰을 결과에서 제외합니다.
    result_gloss = trained_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 'print' 함수와 f-string을 사용하여 최종적으로 생성된 gloss 문자열을 출력합니다.
    print(f"출력 Gloss: {result_gloss}")
    # 'split()' 메소드를 사용하여 생성된 gloss 문자열을 공백 기준으로 나누어 리스트 형태로 출력합니다.
    print(f"출력 Gloss (리스트): {result_gloss.split()}")


# 이 스크립트 파일이 직접 실행될 때만 'main()' 함수를 호출하도록 하는 파이썬의 표준적인 구문입니다.
# 다른 스크립트에서 이 파일을 모듈로 임포트할 경우에는 'main()' 함수가 자동으로 실행되지 않습니다.
if __name__ == '__main__':
    # 'main' 함수를 호출하여 전체 프로세스를 시작합니다.
    data_train()
    # main()