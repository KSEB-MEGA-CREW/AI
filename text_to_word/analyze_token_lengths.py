# Import necessary libraries
import json
from transformers import AutoTokenizer
import numpy as np
import config

# Load the tokenizer from the specified directory
# This is the tokenizer that has been updated with new tokens
NEW_MODEL_DIR = config.NEW_MODEL_DIR
tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL_DIR)

# --- Data Loading ---
# Load the preprocessed data from the JSONL file
# This file contains the data that will be used for training
DATA_PATH = config.DATA_PATH
data = []
# Open the JSONL file and read it line by line
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        # Each line is a JSON object, so we parse it and add to the list
        data.append(json.loads(line))

# Initialize lists to store the lengths of the tokenized inputs and targets
input_lengths = []
target_lengths = []

# Iterate over each item in the dataset
for item in data:
    # Tokenize the source text (koreanText) and store its length
    # The tokenizer converts the text into a sequence of token IDs
    input_ids = tokenizer(item['koreanText'], return_tensors="pt").input_ids
    input_lengths.append(len(input_ids[0]))

    # Join the list of glosses into a single string before tokenizing
    # The gloss is a sequence of words representing the meaning of the sign language
    gloss_text = " ".join(item['gloss_id'])
    target_ids = tokenizer(gloss_text, return_tensors="pt").input_ids
    target_lengths.append(len(target_ids[0]))

# --- Analysis of Token Lengths ---

# Calculate and print statistics for input lengths
print("--- 입력 길이 분석 (Input Length Analysis) ---")
print(f"최소 길이 (Min Length): {np.min(input_lengths)}")
print(f"최대 길이 (Max Length): {np.max(input_lengths)}")
print(f"평균 길이 (Mean Length): {np.mean(input_lengths):.2f}")
print(f"중앙값 (Median Length): {np.median(input_lengths)}")
# Calculate the 95th percentile to understand the distribution of lengths
# This helps in setting a max length that covers most of the data
print(f"95% 백분위수 (95th Percentile): {np.percentile(input_lengths, 95):.2f}")

print("\n" + "="*50 + "\n")

# Calculate and print statistics for target lengths
print("--- 타겟 길이 분석 (Target Length Analysis) ---")
print(f"최소 길이 (Min Length): {np.min(target_lengths)}")
print(f"최대 길이 (Max Length): {np.max(target_lengths)}")
print(f"평균 길이 (Mean Length): {np.mean(target_lengths):.2f}")
print(f"중앙값 (Median Length): {np.median(target_lengths)}")
# Calculate the 95th percentile for the target lengths
# This is useful for setting the max target length for the model
print(f"95% 백분위수 (95th Percentile): {np.percentile(target_lengths, 95):.2f}")

# --- Recommendations for Max Lengths ---

# Provide recommendations for setting max_input_length and max_target_length
# These recommendations are based on the 95th percentile, which is a common practice
print("\n" + "="*50 + "\n")
print("--- 최대 길이 추천 (Max Length Recommendations) ---")
print("max_input_length와 max_target_length는 모델이 처리할 수 있는 최대 토큰 수를 결정합니다.")
print("너무 길게 설정하면 메모리 사용량이 늘어나고, 너무 짧게 설정하면 데이터가 잘릴 수 있습니다.")
print("일반적으로 95% 백분위수를 기준으로 설정하는 것이 좋습니다.")
print(f"추천 max_input_length: {int(np.percentile(input_lengths, 95))}")
print(f"추천 max_target_length: {int(np.percentile(target_lengths, 95))}")