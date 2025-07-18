from datasets import load_dataset

# 사용자의 JSONL 파일 경로를 지정합니다.
data_path = 'path/to/your/data.jsonl'
dataset = load_dataset('json', data_files=data_path)

from transformers import AutoTokenizer

# 사용할 모델의 이름을 지정합니다.
model_checkpoint = "ehekaanldk/kobart2ksl-translation"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    # 입력 텍스트(koreanText)를 토큰화합니다.
    inputs =[]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # 타겟(gloss_id 리스트)을 공백으로 구분된 문자열로 변환한 후 토큰화합니다.
    # 토크나이저의 컨텍스트 매니저를 사용하여 타겟을 올바르게 처리합니다.
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            [' '.join(map(str, gloss_list)) for gloss_list in examples['gloss_id']],
            max_length=128,
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 전처리 함수를 전체 데이터셋에 적용합니다.
tokenized_datasets = dataset.map(preprocess_function, batched=True)