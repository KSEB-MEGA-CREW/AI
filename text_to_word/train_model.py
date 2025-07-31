import warnings
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
    DataCollatorForSeq2Seq,
    # 'EarlyStoppingCallback'은 조기 종료를 위해 사용되는 콜백 클래스입니다.
    EarlyStoppingCallback
)
from accelerate import Accelerator
import util.data_load as data_load
import config
import util.t2g_tokenizer as t2g_tokenizer
import optuna

warnings.filterwarnings("ignore", message="The following device_map keys do not match any submodules in the model:.*")

# 0 --- configuration ---
# 사용할 사전 학습된 KoBART 모델의 이름을 Hugging Face 허브에서 가져와 설정합니다.
MODEL_NAME = config.MODEL_NAME
# 학습에 사용할 데이터 파일(.jsonl)의 경로를 설정합니다.
DATA_PATH = config.DATA_PATH
# 파인튜닝된 모델과 토크나이저, 그리고 학습 로그가 저장될 디렉토리 경로를 설정합니다.
OUTPUT_DIR = config.OUTPUT_DIR
CUSTOM_TOKENIZER = config.CUSTOM_TOKENIZER

# 입력 시퀀스(한국어 문장)의 최대 길이를 128 토큰으로 설정합니다. 이보다 길면 잘라냅니다.
max_input_length = 128
# 타겟 시퀀스(수어 단어)의 최대 길이를 128 토큰으로 설정합니다.
max_target_length = 128


def data_preprocess(dcnt_limit=None, test_size=None, eval_size=None):
    """
    데이터 전처리 함수입니다.
    이 함수는 JSONL 파일에서 데이터를 로드하고, 필요한 형식으로 변환하여 Hugging Face의 Dataset 객체로 반환합니다.
    """
    # 'load_jsonl' 함수를 호출하여 JSONL 파일에서 데이터를 로드합니다.
    train_data = data_load.load_jsonl(DATA_PATH, limit=dcnt_limit)
    
    # 데이터를 딕셔너리 형태로 가공합니다.
    processed_data = {
        "koreanText": [item['koreanText'] for item in train_data],
        "gloss_id": [" ".join(item['gloss_id']) for item in train_data]
    }

    # 가공된 데이터를 Hugging Face의 Dataset 객체로 변환합니다.
    dataset = Dataset.from_dict(processed_data)

    # Load the tokenizer associated with the pre-trained model.
    # The tokenizer is loaded only once and reused across all trials.
    tokenizer = t2g_tokenizer.load_tokenizer()

    # 데이터셋을 모델 입력 형식에 맞게 토큰화하고 변환하는 함수를 정의합니다.
    def preprocess_function(examples):
        # 입력 데이터로 사용할 'koreanText' 컬럼의 텍스트 리스트를 'inputs'에 저장합니다.
        inputs = [ex for ex in examples['koreanText']]
        # 정답(레이블) 데이터로 사용할 'gloss_id' 컬럼의 텍스트 리스트를 'targets'에 저장합니다.
        targets = [ex for ex in examples['gloss_id']]
        
        # 입력 텍스트('inputs')를 토큰화합니다. 'max_length'로 최대 길이를, 'truncation=True'로 길이 초과 시 자르기를, 'padding="max_length"'로 최대 길이에 맞춰 패딩을 설정합니다.
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

        # 'text_target' 인자를 사용하여 타겟(레이블) 텍스트를 토큰화합니다.
        # 이렇게 하면 Seq2Seq 모델의 디코더 입력을 올바르게 처리할 수 있습니다.
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True, padding="max_length")

        # 모델이 정답으로 사용할 수 있도록, 토큰화된 레이블의 'input_ids'를 'model_inputs' 딕셔너리의 "labels" 키에 추가합니다.
        model_inputs["labels"] = labels["input_ids"]
        # 전처리된 'model_inputs' 딕셔너리를 반환합니다.
        return model_inputs

    # 하이퍼파라미터 탐색 시 데이터셋을 훈련용, 검증용으로 분리합니다.
    if test_size is None and eval_size is None:
        tokenized_train_dataset = dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
        print(f"훈련 데이터셋 크기: {len(train_dataset)}")
        return tokenized_train_dataset, None, None

    elif eval_size and test_size is None:
        # 전체 데이터셋을 훈련용(90%)과 검증용(10%)으로 분리합니다. 'test_size=0.1'은 10%를 검증용으로 사용하겠다는 의미입니다.
        dataset = dataset.train_test_split(test_size=eval_size, seed=42)
        # 분리된 데이터셋에서 훈련용 데이터셋을 'train_dataset' 변수에 할당합니다.
        train_dataset = dataset['train']
        # 분리된 데이터셋에서 검증용 데이터셋을 'eval_dataset' 변수에 할당합니다.
        eval_dataset = dataset['test']

        # 'print' 함수와 f-string을 사용하여 훈련 데이터셋의 크기를 출력합니다.
        print(f"훈련 데이터셋 크기: {len(train_dataset)}")
        # 'print' 함수와 f-string을 사용하여 검증 데이터셋의 크기를 출력합니다.
        print(f"검증 데이터셋 크기: {len(eval_dataset)}")

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
        
        return tokenized_train_dataset, tokenized_eval_dataset, None
    else:
        # 전체 데이터셋을 훈련용(90%)과 검증용(10%)으로 분리합니다. 'test_size=0.1'은 10%를 검증용으로 사용하겠다는 의미입니다.
        dataset = dataset.train_test_split(test_size=test_size, seed=42)
        # 분리된 데이터셋에서 훈련용 데이터셋을 'train_dataset' 변수에 할당합니다.
        train_validation_set = dataset['train']
        # 분리된 데이터셋에서 검증용 데이터셋을 'eval_dataset' 변수에 할당합니다.
        test_dataset = dataset['test']

        final_split = train_validation_set.train_test_split(test_size=eval_size/(1-test_size), seed=42)
        train_dataset = final_split['train']
        eval_dataset = final_split['test']

        # 'print' 함수와 f-string을 사용하여 훈련 데이터셋의 크기를 출력합니다.
        print(f"훈련 데이터셋 크기: {len(train_dataset)}")
        # 'print' 함수와 f-string을 사용하여 검증 데이터셋의 크기를 출력합니다.
        print(f"검증 데이터셋 크기: {len(eval_dataset)}")
        print(f"테스트 데이터셋 크기: {len(test_dataset)}")

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
        
        # Tokenize the test dataset before prediction, ensuring it has the same format as the training data.
        tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)
        return tokenized_train_dataset, tokenized_eval_dataset, tokenized_test_dataset




def data_train():

    accelerator = Accelerator()
    # # --- 3. 모델, 토크나이저 로드 ---
    # # 'AutoTokenizer.from_pretrained'를 사용해 'MODEL_NAME'에 해당하는 사전 학습된 모델의 토크나이저를 로드합니다.
    # # A function to initialize the model for hyperparameter search.
    # # This function is called by the Trainer for each trial to get a fresh, untrained model.
    def model_init():
        # Load the pre-trained model specified by MODEL_NAME.
        # This ensures that each hyperparameter search trial starts with the same baseline model.
        # model= t2g_tokenizer.load_model()
        print("Initializing a new model for trial...")
        return t2g_tokenizer.load_model()
    
    # Load the tokenizer associated with the pre-trained model.
    # The tokenizer is loaded only once and reused across all trials.
    model= t2g_tokenizer.load_model()
    tokenizer = t2g_tokenizer.load_tokenizer()

    tokenized_train_dataset, tokenized_eval_dataset, _ = data_preprocess(dcnt_limit=9000, eval_size=0.1)

    # --- 4. Set Training Arguments ---
    # Define various hyperparameters and settings required for model training using the 'Seq2SeqTrainingArguments' class.
    training_args = Seq2SeqTrainingArguments(
        # Specify the directory where training outputs (checkpoints, models, etc.) will be saved.
        output_dir=OUTPUT_DIR,
        # Set the evaluation strategy to be performed at regular step intervals.
        eval_strategy="steps",
        # # Set the number of steps between each evaluation.
        # evaluation_steps=1000,
        # Set the learning rate for the optimizer.
        learning_rate=2e-5,
        # Set the batch size for training on each device (GPU).
        per_device_train_batch_size=16,
        # Set the batch size for evaluation on each device (GPU).
        per_device_eval_batch_size=16,
        # Set the weight decay rate for regularization to prevent overfitting.
        weight_decay=0.01,
        # Limit the total number of saved checkpoints. The oldest ones are deleted first.
        save_total_limit=3,
        # Set the total number of epochs to train the model.
        num_train_epochs=1000,
        # Enable prediction with generation to compute metrics like BLEU and ROUGE during evaluation.
        predict_with_generate=True,
        # Specify the directory for storing logs.
        logging_dir=f'{OUTPUT_DIR}/logs',
        # Set the frequency of logging training information (e.g., loss).
        logging_steps=1000,
        # Set the checkpoint saving strategy to be based on steps.
        save_strategy="steps",
        # Set the number of steps between each checkpoint save. Must be a multiple of evaluation_steps.
        save_steps=1000,
        # Load the best model at the end of training based on the specified metric.
        load_best_model_at_end=True,
        # Specify the metric to use for determining the best model.
        metric_for_best_model="eval_loss",
        # Indicate that a lower value for the metric is better (since it's a loss).
        greater_is_better=False,
        # Set a random seed for reproducibility of training.
        seed=42,
        # Set the maximum gradient norm for gradient clipping to prevent exploding gradients.
        max_grad_norm=1.0,
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
        model_init = model_init, # 모델 초기화 함수 전달
        # 위에서 정의한 학습 관련 인자들을 전달합니다.
        args = training_args,
        # 토큰화된 훈련 데이터셋을 전달합니다.
        train_dataset = tokenized_train_dataset,
        # 토큰화된 검증 데이터셋을 전달합니다.
        eval_dataset = tokenized_eval_dataset,
        # 토크나이저를 전달합니다. (생성된 텍스트를 디코딩하는 데 사용됨)
        tokenizer = tokenizer,
        # 데이터 콜레이터를 전달하여 배치를 구성합니다.
        data_collator=data_collator,
        # 조기 종료 콜백을 추가합니다.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # 'print' 함수로 모델 파인튜닝 시작을 알립니다.
    print("하이퍼파라미터 탐색을 시작합니다.")
    # Optuna의 로깅 레벨을 INFO로 설정하여 각 trial의 상세 정보를 출력합니다.
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # 하이퍼파라미터 탐색 공간을 정의하는 함수입니다.
    def hp_space(trial: optuna.trial.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            # "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16])
        }

    # trainer.hyperparameter_search()를 호출하여 베이지안 최적화를 수행합니다.
    best_run = trainer.hyperparameter_search(
        hp_space=hp_space, # 사용자 정의 탐색 공간 전달
        direction="minimize",  # eval_loss를 최소화하는 것이 목표
        backend="optuna",      # 베이지안 최적화를 위해 optuna 사용
        n_trials=20,           # 20번의 다른 하이퍼파라미터 조합을 시도
        compute_objective=lambda metrics: metrics["eval_loss"], # 최적화 목표 함수
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, # 처음 5개의 trial은 가지치기를 적용하지 않고 끝까지 실행하여 비교 기준이 될 데이터를 축적합니다.
            n_warmup_steps=2, # 각 trial은 최소 2번의 평가(이 코드에서는 2 에포크)를 마친 후에야 가지치기 대상이 됩니다. 초기 학습 단계에서는 성능 변동이 클 수 있기 때문입니다.
            interval_steps=1# 1번의 평가(1 에포크) 주기마다 가지치기 조건을 확인할지를 결정합니다.
            )
    )

    # 최적의 하이퍼파라미터를 출력합니다.
    print("최적의 하이퍼파라미터:", best_run.hyperparameters)

    tokenized_train_dataset, tokenized_eval_dataset, tokenized_test_dataset = data_preprocess(test_size=0.1, eval_size=0.1)
    
    # 조기 종료 patience 값을 변수로 저장
    early_stopping_patience_value = 5
    
    # 전체 데이터 학습
    finally_trainer = Seq2SeqTrainer(
        model_init=model_init, # 모델 초기화 함수 전달
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
        # 조기 종료 콜백을 추가합니다.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience_value)]
    )

    # 최적의 하이퍼파라미터로 모델을 다시 학습합니다.
    for n, v in best_run.hyperparameters.items():
        setattr(finally_trainer.args, n, v)
    
    print("최적의 하이퍼파라미터로 모델 재학습을 시작합니다.")
    train_result = finally_trainer.train()

    # 'print' 함수로 모델 파인튜닝이 완료되었음을 알립니다.
    print("모델 파인튜닝이 완료되었습니다.")

    # 모든 프로세스가 이 지점에 도달할 때까지 기다립니다.
    accelerator.wait_for_everyone()

    print("\n--- 최종 테스트 데이터셋 평가 ---")
    
    # Run prediction on the tokenized test dataset.
    test_results = finally_trainer.predict(tokenized_test_dataset)
    
    # --- 모델 성능 지표 출력 ---
    print("\n" + "="*50)
    print(" " * 15 + "모델 성능 요약")
    print("="*50)

    # 1. 최적 하이퍼파라미터 출력
    print("\n[최적 하이퍼파라미터]")
    for param, value in best_run.hyperparameters.items():
        print(f"- {param}: {value}")

    # 2. 조기 종료 정보 및 학습 에포크
    # log_history에서 eval_loss가 있는 로그만 필터링
    eval_logs = [log for log in finally_trainer.state.log_history if 'eval_loss' in log]
    if eval_logs:
        # eval_loss가 가장 낮은 로그 찾기
        best_eval_log = min(eval_logs, key=lambda x: x['eval_loss'])
        best_epoch = best_eval_log['epoch']
    else:
        best_epoch = 'N/A'

    total_epochs_trained = finally_trainer.state.epoch

    print("\n[학습 정보]")
    print(f"- Early Stopping Patience: {early_stopping_patience_value}")
    print(f"- 조기 종료된 최적 에포크: {best_epoch if isinstance(best_epoch, str) else f'{best_epoch:.2f}'}")
    print(f"- 총 학습된 에포크: {total_epochs_trained:.2f}")

    # 3. Loss 값들 출력
    train_loss = train_result.training_loss
    best_eval_loss = finally_trainer.state.best_metric
    test_loss = test_results.metrics.get('test_loss', 'N/A')

    print("\n[Loss 값]")
    print(f"- 최종 Train Loss: {train_loss:.4f}")
    print(f"- 최적 Eval Loss: {best_eval_loss:.4f}")
    print(f"- 최종 Test Loss: {test_loss if isinstance(test_loss, str) else f'{test_loss:.4f}'}")

    # 4. 전체 테스트 결과
    print("\n[전체 테스트 결과]")
    for key, value in test_results.metrics.items():
        print(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
    
    print("="*50)
    
    # finally_trainer.model은 accelerate에 의해 래핑된 모델일 수 있으므로,
    # .unwrap_model()을 사용하여 원래의 Hugging Face 모델을 가져옵니다.
    unwrapped_model = accelerator.unwrap_model(finally_trainer.model)
    # 메인 프로세스에서만 모델을 저장하도록 하여, 여러 프로세스가 동시에 쓰는 것을 방지합니다.
    if accelerator.is_main_process:
        print(f"학습된 모델을 '{OUTPUT_DIR}'에 저장합니다.")
        # device_map을 사용하지 않고 저장해야, 나중에 device_map='auto'로 불러올 때 유연합니다.
        unwrapped_model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    # -----------------------------------------------------------

# 스크립트의 메인 로직을 포함하는 'main' 함수를 정의합니다.
def main():
    # 10. 추론(Inference) 예시
    # 'print' 함수로 추론 테스트 섹션의 시작을 알립니다.
    print("\n--- 추론 테스트 ---")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"추론시 사용 장치: {device}")

    # 'AutoTokenizer.from_pretrained'를 사용하여 저장된 토크나이저를 다시 불러옵니다.
    trained_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    
    # 'AutoModelForSeq2SeqLM.from_pretrained'를 사용하여 방금 저장한 파인튜닝된 모델을 다시 불러옵니다.
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR, device_map="auto", ignore_mismatched_sizes=True)
    # trained_model.to(device)
    trained_model.eval()

    # 번역(gloss 생성)을 테스트할 한국어 문장을 정의합니다.
    text_to_translate = "안녕하세요 만나서 반갑습니다"
    # 'print' 함수와 f-string으로 어떤 문장이 입력되었는지 출력합니다.
    print(f"입력 문장: {text_to_translate}")

    # 입력 문장을 토큰화합니다. 'return_tensors="pt"'는 결과를 파이토치 텐서로 반환하라는 의미입니다. '.to(trained_model.device)'는 모델이 있는 장치(CPU 또는 GPU)로 텐서를 이동시킵니다.
    inputs = trained_tokenizer(text_to_translate, return_tensors="pt", max_length=max_input_length, truncation=True).to(trained_model.device)
    
    # if there are 'token_type_ids in the inputs, remove that
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
        print(f"input에서 token_type_ids가 제거되었습니다.")

    # 'trained_model.generate()' 메소드를 호출하여 입력 토큰(**inputs)으로부터 새로운 텍스트(gloss)를 생성합니다.
    # outputs = trained_model.generate(**inputs, max_length=max_target_length)
    with torch.no_grad():
        outputs = trained_model.generate(
            **inputs,
            # decoder_start_token_id = trained_tokenizer.eos_token_id,
            num_beams=1,
            do_sample=False,
            max_length=max_target_length
        )
    
    # print(f"generated original token: {outputs}")

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