# evaluation.py
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
# Import the evaluate library to compute metrics.
import evaluate
# Import the tqdm library for progress bars.
from tqdm import tqdm
# Import the glob library to find files matching a specific pattern.
import glob
# Import the json library to parse JSON files.
import json

# Import the sys and os libraries for system-level operations.
import sys
import os
# Get the absolute path of the current script.
__current_path = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory ('text_to_word') to the system path to allow importing modules from it.
sys.path.append(os.path.dirname(__current_path))
# Add the project root directory ('KSEB') to the system path.
sys.path.append(os.path.dirname(os.path.dirname(__current_path)))

# Import the data_load module from the util directory.
from util import data_load
# Import the config module.
import config

# --- Configuration ---
# Set the base directory of the model to be evaluated.
BASE_MODEL_DIR = "/home/202044005/KSEB/text_to_word/kobart-finetuned-ksl-glosser"
# Set the path to the data file.
DATA_PATH = config.DATA_PATH
# Set the maximum length for input sequences.
max_input_length = 128
# Set the maximum length for target sequences.
max_target_length = 128
# Set the size of the test set.
TEST_SIZE = 0.1
# Set the size of the evaluation set.
EVAL_SIZE = 0.1 # This is applied to the remainder

def evaluate_checkpoint(model_dir, tokenizer, test_dataset, epoch=None):
    """
    Evaluate a single model checkpoint.
    
    Args:
        model_dir (str): The directory of the checkpoint to evaluate.
        tokenizer (AutoTokenizer): The tokenizer to use for preprocessing.
        test_dataset (Dataset): The dataset to evaluate on.
        epoch (float, optional): The epoch number of the checkpoint. Defaults to None.
    """
    # Print a separator and the checkpoint being evaluated.
    print("\n" + "="*60)
    print(f"--- Evaluating Checkpoint: {os.path.basename(model_dir)} ---")
    # If the epoch is provided, print it.
    if epoch is not None:
        print(f"--- Epoch: {epoch:.2f} ---")
    # Print the full path of the model directory.
    print(f"--- Model Path: {model_dir} ---")
    # Print another separator.
    print("="*60)

    # Check if CUDA is available and set the device accordingly.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Load the model from the specified directory and move it to the selected device.
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
        # Set the model to evaluation mode.
        model.eval()
    except Exception as e:
        # Print an error message if the model fails to load.
        print(f"Error loading model from {model_dir}: {e}")
        # Continue to the next checkpoint.
        return

    # --- Preprocess Data ---
    def preprocess_function(examples):
        # Get the input texts from the examples.
        inputs = [ex for ex in examples['koreanText']]
        # Get the target texts from the examples.
        targets = [ex for ex in examples['gloss_id']]
        
        # Tokenize the input texts.
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
        # Tokenize the target texts.
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True, padding="max_length")
        
        # Add the tokenized labels to the model inputs.
        model_inputs["labels"] = labels["input_ids"]
        # Return the processed model inputs.
        return model_inputs

    # Tokenize the test dataset.
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

    # --- Generate Predictions ---
    # Print a message indicating that predictions are being generated.
    print("Generating predictions...")
    # Initialize empty lists to store all predictions and labels.
    all_preds = []
    all_labels = []

    # Use a loop with tqdm for a progress bar.
    for batch in tqdm(tokenized_test_dataset.iter(batch_size=16)):
        # Move input tensors to the correct device.
        input_ids = torch.tensor(batch['input_ids']).to(device)
        # Move attention mask tensors to the correct device.
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        # Create a tensor for the labels.
        labels = torch.tensor(batch['labels'])

        # Disable gradient calculation for inference.
        with torch.no_grad():
            # Generate predictions using the model.
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_length,
                num_beams=1,
                do_sample=False
            )

        # Decode the generated predictions into text.
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Replace -100 in labels with the pad_token_id.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode the ground-truth labels into text.
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Add the decoded predictions and labels to their respective lists.
        all_preds.extend(decoded_preds)
        all_labels.extend([[label] for label in decoded_labels])

    # --- Compute and Print Metrics ---
    # Print a message indicating that metrics are being computed.
    print("Computing and printing metrics...")
    try:
        # Load the BLEU, ROUGE, and METEOR metrics.
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
        meteor_metric = evaluate.load("meteor")

        # Compute the scores for each metric.
        bleu_score = bleu_metric.compute(predictions=all_preds, references=all_labels)
        rouge_score = rouge_metric.compute(predictions=all_preds, references=all_labels)
        meteor_score = meteor_metric.compute(predictions=all_preds, references=all_labels)

        # Print the BLEU score details.
        print("\n--- BLEU Score ---")
        print(f"BLEU: {bleu_score['bleu']:.4f}")
        # Print the ROUGE score details.
        print("\n--- ROUGE Score ---")
        print(f"ROUGE-1: {rouge_score['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_score['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
        # Print the METEOR score.
        print("\n--- METEOR Score ---")
        print(f"METEOR: {meteor_score['meteor']:.4f}")

    except Exception as e:
        # Print an error message if metric computation fails.
        print(f"Error computing metrics for {model_dir}: {e}")

def main():
    """
    Main function to find all checkpoints and evaluate them.
    """
    # Print a message indicating the start of the full evaluation process.
    print("--- Starting Full Checkpoint Evaluation ---")

    # --- Load Training Arguments and Data ---
    try:
        # Load the training arguments.
        training_args = torch.load(os.path.join(BASE_MODEL_DIR, "training_args.bin"), weights_only=False)
        # Get the training batch size.
        train_batch_size = training_args.per_device_train_batch_size * training_args._n_gpu
    except Exception as e:
        # Print a warning if the training arguments cannot be loaded.
        print(f"Warning: Could not load training_args.bin: {e}. Epoch numbers will not be calculated.")
        # Set the training batch size to a default value.
        train_batch_size = 16 # Default value

    # Load the full dataset.
    full_data = data_load.load_jsonl(DATA_PATH)
    # Create a Dataset object.
    dataset = Dataset.from_dict({
        "koreanText": [item['koreanText'] for item in full_data],
        "gloss_id": [" ".join(item['gloss_id']) for item in full_data]
    })
    # Split the dataset to get the training and test sets.
    train_validation_split = dataset.train_test_split(test_size=TEST_SIZE, seed=42)
    train_dataset = train_validation_split['train']
    test_dataset = train_validation_split['test']
    # Get the number of training examples.
    num_train_examples = len(train_dataset)
    # Print the number of training and test examples.
    print(f"Loaded {num_train_examples} training examples and {len(test_dataset)} test examples.")

    # --- Find and Evaluate Checkpoints ---
    # Find all checkpoint directories.
    checkpoint_dirs = sorted(glob.glob(os.path.join(BASE_MODEL_DIR, "checkpoint-*")))
    # Add the base model directory to the list of directories to evaluate.
    all_model_dirs = [BASE_MODEL_DIR] + checkpoint_dirs

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

    # Loop through each model directory.
    for model_dir in all_model_dirs:
        # Initialize the epoch to None.
        epoch = None
        # Check if the directory is a checkpoint directory.
        if "checkpoint-" in model_dir:
            try:
                # Extract the step number from the directory name.
                step = int(model_dir.split("-")[-1])
                # Calculate the epoch number.
                epoch = (step * train_batch_size) / num_train_examples
            except (ValueError, IndexError):
                # If parsing fails, the epoch remains None.
                pass
        # Evaluate the checkpoint.
        evaluate_checkpoint(model_dir, tokenizer, test_dataset, epoch=epoch)

    # Print a final message indicating completion.
    print("\n" + "="*60)
    print("All checkpoint evaluations finished.")
    print("="*60)

# Check if the script is being run directly.
if __name__ == "__main__":
    # Call the main function.
    main()