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
import evaluate
import numpy as np
import os
import glob
# Import matplotlib for plotting
# 플롯팅을 위해 matplotlib를 임포트합니다.
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="The following device_map keys do not match any submodules in the model:.*")

# 0 --- configuration ---
# 사용할 사전 학습된 KoBART 모델의 이름을 Hugging Face 허브에서 가져와 설정합니다.
MODEL_NAME = config.MODEL_NAME
# 학습에 사용할 데이터 파일(.jsonl)의 경로를 설정합니다.
DATA_PATH = config.DATA_PATH
# 파인튜닝된 모델과 토크나이저, 그리고 학습 로그가 저장될 디렉토리 경로를 설정합니다.
OUTPUT_DIR = config.OUTPUT_DIR
CUSTOM_TOKENIZER = config.CUSTOM_TOKENIZER
PLOTS_SAVE_DIR = config.PLOTS_SAVE_DIR

# 입력 시퀀스(한국어 문장)의 최대 길이를 128 토큰으로 설정합니다. 이보다 길면 잘라냅니다.
max_input_length = 128
# 타겟 시퀀스(수어 단어)의 최대 길이를 128 토큰으로 설정합니다.
max_target_length = 128

# def save_plots(log_history, all_checkpoint_metrics, output_dir):
#     """
#     Parse the training log history and checkpoint metrics, then save plots.
#     학습 로그 기록과 체크포인트별 지표를 파싱하고 손실 및 평가지표에 대한 플롯을 저장합니다.
#     """
#     # Separate logs for training and evaluation
#     # 학습 및 평가를 위한 로그 분리
#     train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
#     eval_logs = [log for log in log_history if 'eval_loss' in log]

#     # Extract data from logs
#     # 로그에서 데이터 추출
#     train_epochs = [log['epoch'] for log in train_logs] if train_logs else []
#     train_losses = [log['loss'] for log in train_logs] if train_logs else []
    
#     eval_epochs = [log['epoch'] for log in eval_logs] if eval_logs else []
#     eval_losses = [log['eval_loss'] for log in eval_logs] if eval_logs else []
#     eval_meteors = [log.get('eval_meteor', 0) for log in eval_logs] if eval_logs else []
#     eval_bleus = [log.get('eval_bleu', 0) for log in eval_logs] if eval_logs else []
#     eval_rouge1s = [log.get('eval_rouge1', 0) for log in eval_logs] if eval_logs else []

#     # Extract data from checkpoint metrics for test set
#     # 테스트 세트에 대한 체크포인트 메트릭에서 데이터 추출
#     # Ensure metrics are sorted by epoch to draw correct lines
#     test_metrics = sorted([m for m in all_checkpoint_metrics if 'epoch' in m and isinstance(m['epoch'], (int, float))], key=lambda x: x['epoch'])
#     test_epochs = [metric['epoch'] for metric in test_metrics]
#     test_losses = [metric.get('test_loss', 0) for metric in test_metrics]
#     test_meteors = [metric.get('test_meteor', 0) for metric in test_metrics]
#     test_bleus = [metric.get('test_bleu', 0) for metric in test_metrics]
#     test_rouge1s = [metric.get('test_rouge1', 0) for metric in test_metrics]

#     # --- Plot 1: Training, Validation, and Test Loss ---
#     # --- 플롯 1: 학습, 검증, 및 테스트 손실 ---
#     plt.figure(figsize=(12, 6))
#     if train_epochs:
#         plt.plot(train_epochs, train_losses, label='Training Loss', marker='o', linestyle='--')
#     if eval_epochs:
#         plt.plot(eval_epochs, eval_losses, label='Validation Loss', marker='o', linestyle='-')
#     if test_epochs:
#         plt.plot(test_epochs, test_losses, label='Test Loss', marker='s', linestyle=':')
#     plt.title('Training, Validation & Test Loss Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     if train_epochs or eval_epochs or test_epochs:
#         plt.legend()
#     plt.grid(True)
#     loss_plot_path = os.path.join(output_dir, 'training_validation_test_loss.png')
#     plt.savefig(loss_plot_path)
#     print(f"Loss plot saved to {loss_plot_path}")
#     plt.close()

#     # --- Plot 2: Evaluation and Test Metrics (METEOR, BLEU, ROUGE) ---
#     # --- 플롯 2: 검증 및 테스트 지표 (METEOR, BLEU, ROUGE) ---
#     plt.figure(figsize=(15, 8))
#     # Plot validation metrics if available
#     if eval_epochs:
#         plt.plot(eval_epochs, eval_meteors, label='Validation METEOR', color='green', marker='o', linestyle='-')
#         plt.plot(eval_epochs, eval_bleus, label='Validation BLEU', color='blue', marker='o', linestyle='-')
#         plt.plot(eval_epochs, eval_rouge1s, label='Validation ROUGE-1', color='red', marker='o', linestyle='-')
#     # Plot test metrics if available
#     if test_epochs:
#         plt.plot(test_epochs, test_meteors, label='Test METEOR', color='limegreen', marker='s', linestyle='--')
#         plt.plot(test_epochs, test_bleus, label='Test BLEU', color='deepskyblue', marker='s', linestyle='--')
#         plt.plot(test_epochs, test_rouge1s, label='Test ROUGE-1', color='tomato', marker='s', linestyle='--')
    
#     plt.title('Evaluation & Test Metrics Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Score')
#     if eval_epochs or test_epochs:
#         plt.legend()
#     plt.grid(True)
#     metrics_plot_path = os.path.join(output_dir, 'evaluation_test_metrics.png')
#     plt.savefig(metrics_plot_path)
#     print(f"Metrics plot saved to {metrics_plot_path}")
#     plt.close()

def save_plots(log_history, all_checkpoint_metrics, output_dir):
    """
    Parse the training log history and checkpoint metrics, then save plots.
    This version creates 5 separate plots: Loss, BLEU, ROUGE, METEOR, and a combined metrics plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Separate logs for training and evaluation
    train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log]

    # Extract data from logs for plotting
    train_epochs = [log['epoch'] for log in train_logs]
    train_losses = [log['loss'] for log in train_logs]
    
    eval_epochs = [log['epoch'] for log in eval_logs]
    eval_losses = [log['eval_loss'] for log in eval_logs]
    eval_meteors = [log.get('eval_meteor') for log in eval_logs]
    eval_bleus = [log.get('eval_bleu') for log in eval_logs]
    eval_rouge1s = [log.get('eval_rouge1') for log in eval_logs]

    # Colors for checkpoint lines
    checkpoint_colors = ['red', 'green', 'purple']

    # --- Plot 1: Loss Plot ---
    plt.figure(figsize=(12, 8))
    # Plot training loss over epochs
    plt.plot(train_epochs, train_losses, label='Training Loss',  linestyle='--')
    # Plot validation loss over epochs
    plt.plot(eval_epochs, eval_losses, label='Validation Loss', marker='o', linestyle='-')
    
    # Plot test loss for each checkpoint as a horizontal line
    for i, metrics in enumerate(all_checkpoint_metrics):
        if 'test_loss' in metrics:
            # Add a horizontal line for the test loss of the current checkpoint
            plt.axhline(y=metrics['test_loss'], color=checkpoint_colors[i % len(checkpoint_colors)], linestyle=':', label=f"Test Loss ({metrics.get('checkpoint', 'N/A')})")

    # Set plot title
    plt.title('Loss Over Epochs')
    # Set x-axis label
    plt.xlabel('Epoch')
    # Set y-axis label
    plt.ylabel('Loss')
    # Display the legend
    plt.legend()
    # Enable the grid
    plt.grid(True)
    # Define the save path for the plot
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    # Save the figure
    plt.savefig(loss_plot_path)
    # Announce the save location
    print(f"Loss plot saved to {loss_plot_path}")
    # Close the plot to free memory
    plt.close()

    # --- Plot 2: BLEU Score Plot ---
    plt.figure(figsize=(12, 8))
    # Note: Training BLEU scores are not logged, only evaluation scores are available.
    if any(b is not None for b in eval_bleus):
        # Plot validation BLEU score over epochs
        plt.plot(eval_epochs, eval_bleus, label='Validation BLEU', marker='o', linestyle='-')

    # Plot test BLEU score for each checkpoint as a horizontal line
    for i, metrics in enumerate(all_checkpoint_metrics):
        if 'test_bleu' in metrics:
            # Add a horizontal line for the test BLEU score
            plt.axhline(y=metrics['test_bleu'], color=checkpoint_colors[i % len(checkpoint_colors)], linestyle=':', label=f"Test BLEU ({metrics.get('checkpoint', 'N/A')})")

    # Set plot title
    plt.title('BLEU Score Over Epochs')
    # Set x-axis label
    plt.xlabel('Epoch')
    # Set y-axis label
    plt.ylabel('BLEU Score')
    # Display the legend
    plt.legend()
    # Enable the grid
    plt.grid(True)
    # Define the save path for the plot
    bleu_plot_path = os.path.join(output_dir, 'bleu_plot.png')
    # Save the figure
    plt.savefig(bleu_plot_path)
    # Announce the save location
    print(f"BLEU plot saved to {bleu_plot_path}")
    # Close the plot
    plt.close()

    # --- Plot 3: ROUGE Score Plot ---
    plt.figure(figsize=(12, 8))
    # Note: Training ROUGE scores are not logged, only evaluation scores are available.
    if any(r is not None for r in eval_rouge1s):
        # Plot validation ROUGE-1 score over epochs
        plt.plot(eval_epochs, eval_rouge1s, label='Validation ROUGE-1', marker='o', linestyle='-')

    # Plot test ROUGE score for each checkpoint as a horizontal line
    for i, metrics in enumerate(all_checkpoint_metrics):
        if 'test_rouge1' in metrics:
            # Add a horizontal line for the test ROUGE-1 score
            plt.axhline(y=metrics['test_rouge1'], color=checkpoint_colors[i % len(checkpoint_colors)], linestyle=':', label=f"Test ROUGE-1 ({metrics.get('checkpoint', 'N/A')})")

    # Set plot title
    plt.title('ROUGE-1 Score Over Epochs')
    # Set x-axis label
    plt.xlabel('Epoch')
    # Set y-axis label
    plt.ylabel('ROUGE-1 Score')
    # Display the legend
    plt.legend()
    # Enable the grid
    plt.grid(True)
    # Define the save path for the plot
    rouge_plot_path = os.path.join(output_dir, 'rouge_plot.png')
    # Save the figure
    plt.savefig(rouge_plot_path)
    # Announce the save location
    print(f"ROUGE plot saved to {rouge_plot_path}")
    # Close the plot
    plt.close()

    # --- Plot 4: METEOR Score Plot ---
    plt.figure(figsize=(12, 8))
    # Note: Training METEOR scores are not logged, only evaluation scores are available.
    if any(m is not None for m in eval_meteors):
        # Plot validation METEOR score over epochs
        plt.plot(eval_epochs, eval_meteors, label='Validation METEOR', marker='o', linestyle='-')

    # Plot test METEOR score for each checkpoint as a horizontal line
    for i, metrics in enumerate(all_checkpoint_metrics):
        if 'test_meteor' in metrics:
            # Add a horizontal line for the test METEOR score
            plt.axhline(y=metrics['test_meteor'], color=checkpoint_colors[i % len(checkpoint_colors)], linestyle=':', label=f"Test METEOR ({metrics.get('checkpoint', 'N/A')})")

    # Set plot title
    plt.title('METEOR Score Over Epochs')
    # Set x-axis label
    plt.xlabel('Epoch')
    # Set y-axis label
    plt.ylabel('METEOR Score')
    # Display the legend
    plt.legend()
    # Enable the grid
    plt.grid(True)
    # Define the save path for the plot
    meteor_plot_path = os.path.join(output_dir, 'meteor_plot.png')
    # Save the figure
    plt.savefig(meteor_plot_path)
    # Announce the save location
    print(f"METEOR plot saved to {meteor_plot_path}")
    # Close the plot
    plt.close()

    # --- Plot 5: Combined Metrics Plot (BLEU, ROUGE, METEOR) ---
    plt.figure(figsize=(15, 10))
    # Plot evaluation metrics if they are available
    if any(b is not None for b in eval_bleus):
        plt.plot(eval_epochs, eval_bleus, label='Validation BLEU', marker='o', linestyle='-', color='blue')
    if any(r is not None for r in eval_rouge1s):
        plt.plot(eval_epochs, eval_rouge1s, label='Validation ROUGE-1', marker='o', linestyle='-', color='red')
    if any(m is not None for m in eval_meteors):
        plt.plot(eval_epochs, eval_meteors, label='Validation METEOR', marker='o', linestyle='-', color='green')

    # # Plot test metrics for each checkpoint as horizontal lines
    # for i, metrics in enumerate(all_checkpoint_metrics):
    #     # Get the checkpoint name for the legend
    #     checkpoint_name = metrics.get('checkpoint', 'N/A')
    #     # Add horizontal line for test BLEU score
    #     if 'test_bleu' in metrics:
    #         plt.axhline(y=metrics['test_bleu'], linestyle=':', color='cyan', label=f"Test BLEU ({checkpoint_name})")
    #     # Add horizontal line for test ROUGE-1 score
    #     if 'test_rouge1' in metrics:
    #         plt.axhline(y=metrics['test_rouge1'], linestyle=':', color='magenta', label=f"Test ROUGE-1 ({checkpoint_name})")
    #     # Add horizontal line for test METEOR score
    #     if 'test_meteor' in metrics:
    #         plt.axhline(y=metrics['test_meteor'], linestyle=':', color='lime', label=f"Test METEOR ({checkpoint_name})")

    # Set plot title
    plt.title('Combined Evaluation Metrics Over Epochs')
    # Set x-axis label
    plt.xlabel('Epoch')
    # Set y-axis label
    plt.ylabel('Score')
    # Display the legend in the best location
    plt.legend(loc='best')
    # Enable the grid
    plt.grid(True)
    # Define the save path for the plot
    combined_metrics_plot_path = os.path.join(output_dir, 'combined_metrics_plot.png')
    # Save the figure
    plt.savefig(combined_metrics_plot_path)
    # Announce the save location
    print(f"Combined metrics plot saved to {combined_metrics_plot_path}")
    # Close the plot
    plt.close()

# 데이터 전처리 함수
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
        print(f"훈련 데이터셋 크기: {len(dataset)}")
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

# 학습을 수행하는 함수
# the function of training the model
def data_train():

    accelerator = Accelerator()
    
    # Load the tokenizer associated with the pre-trained model.
    # The tokenizer is loaded only once and reused across all trials.
    model= t2g_tokenizer.load_model()
    tokenizer = t2g_tokenizer.load_tokenizer()

    # Load the metric for evaluation
    # Meteor, BLEU, ROUGE 평가 지표를 계산할 수 있는 객체를 불러옵니다.
    meteor_metric = evaluate.load("meteor")
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        """
        Compute METEOR, BLEU, and ROUGE scores by comparing model predictions to the ground truth.
        모델의 예측 결과와 실제 정답을 비교하여 METEOR, BLEU, ROUGE 지표를 계산하는 함수입니다.
        """
        # Unpack predictions and labels from the EvalPrediction object.
        preds, labels = eval_preds
        
        # Check if predictions are in a tuple and extract the primary output if so.
        # 예측이 튜플 인지 확인하고, 그렇다면 기본 출력인 첫번째 요소 추출.
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Decode the generated token IDs back to text.
        # Replace -100 (used for padding) with the pad_token_id before decoding.
        # 디코딩을 하기 전에 -100(padding에 사용한)을 pad_token_id로 대체합니다. 
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Decode the ground-truth label IDs back to text.
        # Replace -100 with pad_token_id for labels.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode the label IDs to strings.
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # To prevent a ZeroDivisionError in the BLEU score calculation,
        # if a prediction is empty, we replace it with a placeholder.
        # OLD CODE:
        # decoded_preds = [pred if pred.strip() else " " for pred in decoded_preds]
        
        # To prevent ZeroDivisionError, replace empty predictions with a placeholder token.
        # A single space might be tokenized as empty, so we use a specific token like '[PAD]'.
        decoded_preds = [pred.strip() if pred.strip() else "[PAD]" for pred in decoded_preds]
        # Also, ensure labels are not empty, which could cause issues with some metrics.
        decoded_labels = [label.strip() if label.strip() else "[PAD]" for label in decoded_labels]

        # Prepare references for metrics - they expect a list of lists.
        decoded_labels_for_metric = [[label] for label in decoded_labels]

        # Compute the scores.
        # Calculate the METEOR score.
        meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels_for_metric)
        # Calculate the BLEU score.
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_for_metric)
        # Calculate the ROUGE score.
        rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels_for_metric)
        
        # Combine all metrics into a single dictionary.
        result = {
            "meteor": meteor_result["meteor"],
            "bleu": bleu_result["bleu"]
        }
        result.update(rouge_result) # ROUGE returns multiple scores (rouge1, rouge2, etc.)
        
        # Add generated text length to the metrics for analysis.
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    # A function to initialize the model for hyperparameter search.
    def model_init():
        # Load the pre-trained model specified by MODEL_NAME.
        print("Initializing a new model for trial...")
        return t2g_tokenizer.load_model()

    #*******************************************************************
    #*******************************************************************
    # --1-- 데이터 로드
    # Load and preprocess the dataset.
    tokenized_train_dataset, tokenized_eval_dataset, _ = data_preprocess(dcnt_limit=9000, eval_size=0.1)
    #*******************************************************************
    #*******************************************************************

    # --2-- 모델 초기화
    # Initialize the model using Seq2SeqTraingingArguments function 
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # Set evaluation strategy to "epoch" to evaluate at the end of each epoch.
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        # Enable prediction with generation to compute metrics like METEOR.
        predict_with_generate=True,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=5,
        save_strategy="epoch",
        # Load the best model at the end based on the METEOR score.
        load_best_model_at_end=True,
        # Specify "meteor" as the metric for finding the best model.
        metric_for_best_model="meteor",
        # Indicate that a higher METEOR score is better.
        greater_is_better=True,
        seed=42,
        max_grad_norm=1.0,
    )

    # --- Define Data Collator ---
    #  data_collator는 데이터셋의 개별 샘플들을 가져와 모델이 학습할 수 있는 형태의 '미니 배치(mini-batch)'로 효율적이고 올바르게 조립해주는 역할을 합니다. 
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model
    )
    

    # --- Initialize Trainer ---
    # 모델의 학습을 수행하기 위해 Seq2SeqTrainer를 초기화합니다.
    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # Pass the compute_metrics function to the trainer.
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience_value)]
    )

    print("\n" + "="*50)
    print(" " * 15 + "Starting hyperparameter search.")
    print(" " * 15 + "하이퍼파라미터 탐색 시작")
    print("="*50)

    # --- Hyperparameter Search ---
    optuna.logging.set_verbosity(optuna.logging.INFO)

    def hp_space(trial: optuna.trial.Trial):
        """
        Define the hyperparameter search space for Optuna.
        Optuna를 위한 하이퍼파라미터 탐색 공간을 정의합니다.
        """
        print(f"\n--- Starting Trial {trial.number} ---")
        print("  - Suggested hyperparameters:")
        batch_size = trial.suggest_categorical("per_device_batch_size", [16, 32])

        # Suggest and define hyperparameters in a dictionary
        # 하이퍼파라미터를 제안하고 딕셔너리로 정의합니다.
        # 
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 50, 100, step=10),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "generation_num_beams": trial.suggest_int( "generation_num_beams", 3, 10)
        }

        for key, value in params.items():
            print(f"    - {key}: {value}")

        return params

    # The objective for hyperparameter search is now to maximize the METEOR score.
    best_run = trainer.hyperparameter_search(
        hp_space=hp_space,
        direction="maximize",  # Maximize the METEOR score
        backend="optuna",
        n_trials=50,
        compute_objective=lambda metrics: metrics["eval_meteor"], # Objective is eval_meteor
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=3
        )
    )

    print("--Optimal parameters--")
    print("최적의 파라미터:")

    #*******************************************************************
    #*******************************************************************
    # tokenized_train_dataset, tokenized_eval_dataset, tokenized_test_dataset = data_preprocess(dcnt_limit=9000, test_size=0.1, eval_size=0.1)
    tokenized_train_dataset, tokenized_eval_dataset, tokenized_test_dataset = data_preprocess(test_size=0.1, eval_size=0.1)
    #*******************************************************************
    #*******************************************************************

    # --- Final Training with Best Hyperparameters ---
    # Re-initialize the model to train all data with aptimal hyperparameters.
    # 최적의 하이퍼파라미터로 모델이 모든 데이터를 학습 위해 다시 초기화 합니다. 
    final_trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience_value)]
    )

    # Set the best hyperparameters found during the search.
    for n, v in best_run.hyperparameters.items():
        # v += 5 if n == "num_train_epochs" else v
        print(n + ": " + str(v))
        setattr(final_trainer.args, n, v)

    print("\n최적의 하이퍼파라미터로 모델 재학습을 시작합니다.\n")
    train_result = final_trainer.train()

    print("\n모델 파인튜닝이 완료되었습니다\n")
    accelerator.wait_for_everyone()

    # --- Final Evaluation of the Best Model ---
    print("\n--- 최종 테스트 데이터셋 평가 (최적 모델) ---")
    test_results = final_trainer.predict(tokenized_test_dataset)
    
    # --- Print Model Performance Summary ---
    print("\n" + "="*50)
    print(" " * 15 +"모델 성능 요약")
    print("="*50)

    print("\n[최적 하이퍼파라미터]")
    for param, value in best_run.hyperparameters.items():
        print(f"- {param}: {value}")

    # Find the best epoch based on the highest METEOR score
    # 최종 학습 모델의 로그(final_trainer.state.log_history)에서 각 요소 중 'eval_meteor' 키가 있는 로그만 가져와 배열에 저장합니다.
    # From the logs of the final training model (final_trainer.state.log_history), fetch only the logs where each element has the key “eval_meteor” and store them in an array.
    eval_logs = [log for log in final_trainer.state.log_history if 'eval_meteor' in log]
    if eval_logs:
        # Find the log with the highest value based on the 'eval_meteor' key in eval_logs, the list that holds the evaluation elements.
        # 평가 요소를 보관하는 리스트인 eval_logs에서  'eval_meteor' 키를 기준으로 가장 높은 값을 가진 로그를 찾습니다. 
        best_eval_log = max(eval_logs, key=lambda x: x['eval_meteor'])
        best_epoch = best_eval_log['epoch']
        
        print("\n[체크포인트별 검증(Validation) 결과]")
        # Loop through the evaluation logs and print results for each checkpoint
        for log in eval_logs:
            # Check if 'epoch' and other metrics are in the log
            if 'epoch' in log and 'eval_loss' in log and 'eval_meteor' in log:
                # Print the formatted results for the current epoch
                metric_str = f"Eval Loss = {log['eval_loss']:.4f}, Eval METEOR = {log['eval_meteor']:.4f}"
                if 'eval_bleu' in log:
                    metric_str += f", Eval BLEU = {log['eval_bleu']:.4f}"
                if 'eval_rouge1' in log:
                    metric_str += f", Eval ROUGE-1 = {log['eval_rouge1']:.4f}"
                print(f"  - Epoch {log['epoch']:.2f}: {metric_str}")
    else:
        best_epoch = 'N/A'


    total_epochs_trained = final_trainer.state.epoch

    print("\n[학습 정보]")
    # print(f"- Early Stopping Patience: {early_stopping_patience_value}")
    # print(f"- 조기 종료된 최적 에포크: {best_epoch if isinstance(best_epoch, str) else f'{best_epoch:.2f}'}")
    print(f"- 총 학습된 에포크: {total_epochs_trained:.2f}")

    # --- Evaluate All Saved Checkpoints on Test Set ---
    print("\n--- 저장된 모든 체크포인트에 대한 테스트 데이터셋 평가 ---")
    checkpoint_dirs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    
    all_checkpoint_metrics = []

    # Iterate through the checkpoint directories and evaluate each checkpoint.
    # 체크포인트 디렉터리를 순회하며 각 체크포인트에 대한 평가를 진행합니다.
    for checkpoint_dir in checkpoint_dirs:
        # Print the start of evaluation for the current checkpoint. 
        print(f"\n--- {os.path.basename(checkpoint_dir)} 평가 시작 ---")
        
        # Initialize epoch_info to 'N/A' as a default value.
        # epoch 정보를 'N/A'로 초기화 합니다.
        epoch_info = "N/A"

        # Start a try-except block to handle potential errors gracefully.
        # 잠재적인 오류를 처리하기 위한 try_except 블록
        try:
            # Extract the step number from the checkpoint directory name (e.g., 'checkpoint-56007' -> 56007).
            # 디렉터리 이름에서 체크포인트의 step number를 추출합니다.
            checkpoint_step = int(os.path.basename(checkpoint_dir).split('-')[-1])
            # Iterate through the log history of the final trainer.
            # final_trainer의 로그를 추출합니다.
            # This is to find the epoch number corresponding to the current checkpoint.
            # 현재 체크포인트에 해당하는 epoch 번호를 찾기 위함입니다.
            for log in final_trainer.state.log_history:
                # Check if the log entry corresponds to the current checkpoint's step and contains an 'epoch' key.
                # 로그 항목이 현제 체크포인트와 일치한지 확인하고 'epoch' 키가 있는지 확인
                if log.get('step') == checkpoint_step and 'epoch' in log:
                    # If found, format the epoch number to two decimal places and store it.
                    # 만약 찾았다면, epoch 번호를 포멧하여 저장합니다.
                    epoch_info = f"{log['epoch']:.2f}"
                    # Exit the loop once the matching log is found.
                    break
        # Catch exceptions if the directory name is not in the expected format.
        except (ValueError, IndexError):
            # If an error occurs, epoch_info remains 'N/A'.
            pass

        # Load the model from the specific checkpoint directory.

        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir, ignore_mismatched_sizes=True)
        
        # Create a temporary set of training arguments specifically for prediction.
        # 예측을 위해 임시적인 훈련 인자 집합을 생성합니다.
        temp_training_args = Seq2SeqTrainingArguments(
            # Use the same output directory as the main training.
            output_dir=training_args.output_dir,
            # Enable prediction using the generate method, which is necessary for text generation tasks.
            predict_with_generate=True,
            # Set evaluation strategy to "no" to prevent the trainer from requiring an eval_dataset.
            # evaluation strategy를 "no"로 설정하여 trainer가 eval_dataset을 요구하지 않도록 합니다.
            eval_strategy="no",
            # Use the same evaluation batch size as the main training for consistency.
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        )
        
        # Initialize a temporary Seq2SeqTrainer for running predictions.
        # 임시 Seq2SeqTrainer를 초기화하여 예측을 실행합니다.
        temp_trainer = Seq2SeqTrainer(
            # Pass the loaded model from the checkpoint.
            model=model,
            # Use the temporary training arguments.
            args=temp_training_args,
            # Pass the tokenizer.
            tokenizer=tokenizer,
            # Pass the data collator.
            data_collator=data_collator,
            # Pass the function to compute metrics.
            compute_metrics=compute_metrics,
        )
        
        # Run prediction on the test dataset using the temporary trainer.
        checkpoint_test_results = temp_trainer.predict(tokenized_test_dataset)
        # Extract the metrics from the prediction results.
        metrics = checkpoint_test_results.metrics
        # Add the checkpoint directory name to the metrics for identification.
        metrics["checkpoint"] = os.path.basename(checkpoint_dir)
        metrics["epoch"] = epoch_info
        # Append the metrics for this checkpoint to the list of all metrics.
        all_checkpoint_metrics.append(metrics)
        
        print("\n" + "="*50)
        # Print the header for the evaluation results of the current checkpoint.
        print(f"--- {os.path.basename(checkpoint_dir)} 평가 결과 ---")
        print("="*50)
        # Print the epoch number associated with this checkpoint.
        print(f"- Epoch: {epoch_info}")
        # Iterate through the computed metrics.
        for key, value in metrics.items():
            # Do not print the 'checkpoint' key again as it's already in the header.
            if key != "checkpoint":
                # Print each metric's key and value, formatted to 4 decimal places if it's a float.
                print(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")

    if all_checkpoint_metrics:
            print("\n[체크포인트별 테스트 성능 요약]")
            for metrics in all_checkpoint_metrics:
                checkpoint_name = metrics.get("checkpoint", "N/A")
                epoch_val = metrics.get("epoch", "N/A")
                # Format other metrics for printing
                metric_str = ", ".join([f"{k.replace('test_', '')}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
                print(f"  - {checkpoint_name} (Epoch: {epoch_val}): {metric_str}")

    # --- Save Plots ---
    # Save the training and evaluation plots to the output directory.
    # 학습 및 평가 플롯을 출력 디렉터리에 저장합니다.
    save_plots(final_trainer.state.log_history, all_checkpoint_metrics, PLOTS_SAVE_DIR)

    train_loss = train_result.training_loss
    
    print("\n[최적 체크포인트 성능 요약 (검증 데이터 기준)]")
    print(f"- 총 학습 Train Loss: {train_loss:.4f}")

    if eval_logs:
        # Find the training log closest to the best epoch

        train_logs = [log for log in final_trainer.state.log_history if 'loss' in log and 'eval_loss' not in log]
        train_loss_at_best_epoch = 'N/A'
        if train_logs:
            # 최적의 epoch에 가장 가까운 학습 로그를 찾습니다.
            # Find the training log closest to the best epoch
            closest_train_log = min(train_logs, key=lambda x: abs(x['epoch'] - best_epoch))
            train_loss_at_best_epoch = closest_train_log.get('loss')

        eval_loss_at_best_epoch = best_eval_log.get('eval_loss')

        print(f"\n--- 최적 에포크({best_epoch:.2f}) 성능 ---")
        if isinstance(train_loss_at_best_epoch, float):
            print(f"- Train Loss: {train_loss_at_best_epoch:.4f}")
        if eval_loss_at_best_epoch is not None:
            print(f"- Eval Loss: {eval_loss_at_best_epoch:.4f}")
        if 'eval_meteor' in best_eval_log:
            print(f"- Eval METEOR: {best_eval_log['eval_meteor']:.4f}")
        if 'eval_bleu' in best_eval_log:
            print(f"- Eval BLEU: {best_eval_log['eval_bleu']:.4f}")
        if 'eval_rouge1' in best_eval_log:
            print(f"- Eval ROUGE-1: {best_eval_log['eval_rouge1']:.4f}")


    print("\n--- 최종 테스트 성능 (최적 모델) ---")
    print("[전체 테스트 결과 상세]")
    for key, value in test_results.metrics.items():
        print(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
    
    print("="*50)
    
    unwrapped_model = accelerator.unwrap_model(final_trainer.model)
    if accelerator.is_main_process:
        print(f"Saving the trained model to '{OUTPUT_DIR}'")
        unwrapped_model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    # -----------------------------------------------------------

# 이 스크립트 파일이 직접 실행될 때만 'main()' 함수를 호출하도록 하는 파이썬의 표준적인 구문입니다.
# 다른 스크립트에서 이 파일을 모듈로 임포트할 경우에는 'main()' 함수가 자동으로 실행되지 않습니다.
if __name__ == '__main__':
    # 'main' 함수를 호출하여 전체 프로세스를 시작합니다.
    data_train()
