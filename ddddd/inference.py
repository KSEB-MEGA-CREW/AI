import warnings
import torch

# This file is now a clean module that only contains the inference logic.
# All pipeline initialization has been moved to app.py to ensure correct startup order.

def inference(input_text: str, text_to_gloss_pipeline: object):
    """
    Generates a sequence of glosses from an input Korean sentence using the provided pipeline.
    
    Args:
        input_text: The Korean sentence to be translated into glosses.
        text_to_gloss_pipeline: The initialized Hugging Face pipeline object.
        
    Returns:
        A list of gloss strings.
    """
    # Print the input sentence for logging purposes
    print(f"\nInput text: {input_text}")

    # Use a torch.no_grad() context manager for inference to disable gradient calculations,
    # which saves memory and speeds up computation.
    with torch.no_grad():
        # Call the pipeline object with the input text.
        # Pass generation parameters like num_beams and max_length directly.
        results = text_to_gloss_pipeline(
            input_text,
            num_beams=8,
            do_sample=False,
            max_length=128
        )
    
    # The pipeline returns a list of dictionaries, e.g., [{'translation_text': '...'}].
    # Extract the translated text from the first element.
    result_gloss = results[0]['translation_text']
    
    # Print the final generated gloss string and the list version.
    print(f"Output gloss: {result_gloss}")
    # Split the gloss string into a list of individual glosses.
    gloss_list = result_gloss.split()
    # Print the list of glosses.
    print(f"Output gloss (List): {gloss_list}")
    
    # Return the list of glosses.
    return gloss_list