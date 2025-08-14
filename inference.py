import warnings
import torch
from transformers import pipeline # Import pipeline
import config

# Suppress specific warnings for a cleaner output
warnings.filterwarnings("ignore", message="The following device_map keys do not match any submodules in the model:.*")

# --- Configuration ---
# Get the output directory for the model from the T2G_flask/config.py file
OUTPUT_DIR = config.OUTPUT_DIR
# Set the maximum length for the target sequence (gloss)
max_target_length = 128

# --- Pipeline Initialization ---
# Create the pipeline object once when the script is loaded.
# This is much more efficient as the model is loaded into memory only once.
print("Initializing Text-to-Gloss pipeline for T2G_flask...")
# Define the pipeline with the appropriate task, model path, and tokenizer path.
# device_map='auto' will automatically use a GPU if available.
text_to_gloss_pipeline = pipeline(
    "translation_ko_to_KSL",  # This task is suitable for sequence-to-sequence models.
    model=OUTPUT_DIR,
    tokenizer=OUTPUT_DIR,
    device_map="auto"
)
print("Pipeline initialized successfully.")

def inference(input_text: str, text_to_gloss_pipeline: object):
    """
    Generates a sequence of glosses from an input Korean sentence using the initialized pipeline.
    
    Args:
        input_text: The Korean sentence to be translated into glosses.
        
    Returns:
        A list of gloss strings.
    """
    # Print the input sentence for logging purposes
    print(f"\n입력 문자: {input_text}")

    # Use a torch.no_grad() context manager for inference to disable gradient calculations,
    # which saves memory and speeds up computation.
    with torch.no_grad():
        # Call the pipeline object with the input text.
        # Pass generation parameters like num_beams and max_length directly.
        results = text_to_gloss_pipeline(
            input_text,
            num_beams=8,
            do_sample=False,
            max_length=max_target_length
        )
    
    # The pipeline returns a list of dictionaries, e.g., [{'translation_text': '...'}].
    # Extract the translated text from the first element.
    result_gloss = results[0]['translation_text']
    
    # Print the final generated gloss string and the list version.
    print(f"출력 글로스: {result_gloss}")
    # Split the gloss string into a list of individual glosses.
    gloss_list = result_gloss.split()
    # Print the list of glosses.
    print(f"출력 글로스 (List): {gloss_list}")
    
    # Return the list of glosses.
    return gloss_list


# This standard Python construct ensures that the following code runs only when
# the script is executed directly (not when imported as a module).
if __name__ == '__main__':
    # Define a test sentence.
    test_sentence = "사춘기 때 아이에게 일어나는 변화를 잘 이해하고 지나가는 것이 필요합니다."
    # Call the inference function to start the process.
    inference(test_sentence, text_to_gloss_pipeline)
