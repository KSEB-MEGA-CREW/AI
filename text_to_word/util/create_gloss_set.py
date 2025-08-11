
import json
# Import the json module to handle JSON data.

def create_gloss_set():
    # Define a function to create a set of glosses.
    
    input_file_path = '/home/202044005/KSEB/text_to_word/preprocessed_data/processed_data.jsonl'
    # Set the path to the input JSONL file.
    
    output_file_path = '/home/202044005/KSEB/text_to_word/preprocessed_data/unique_glosses.json'
    # Set the path for the output JSON file.
    
    gloss_set = set()
    # Create an empty set to store unique glosses.
    
    try:
        # Start a try block to handle potential file errors.
        with open(input_file_path, 'r', encoding='utf-8') as f:
            # Open the input file for reading with UTF-8 encoding.
            for line in f:
                # Iterate over each line in the file.
                try:
                    # Start a nested try block for JSON parsing.
                    data = json.loads(line)
                    # Parse the JSON data from the current line.
                    if 'gloss_id' in data:
                        # Check if the 'gloss' key exists in the data.
                        gloss_ids = data['gloss_id']
                        cleaned_glosses = [''.join(g.split()) for g in gloss_ids]
                        gloss_set.update(cleaned_glosses)
                        # gloss_set.update(data['gloss_id'])
                        # Add the value of the 'gloss' key to the set.
                except json.JSONDecodeError:
                    # Handle cases where a line is not valid JSON.
                    print(f"Warning: Could not decode JSON from line: {line.strip()}")
                    # Print a warning message for the invalid line.
                    
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Open the output file for writing with UTF-8 encoding.
            json.dump(list(gloss_set), f, ensure_ascii=False, indent=4)
            # Write the list of unique glosses to the JSON file with indentation.
            
        print(f"Successfully saved unique glosses to {output_file_path}")
        # Print a success message after saving the file.
        
    except FileNotFoundError:
        # Handle the case where the input file is not found.
        print(f"Error: Input file not found at {input_file_path}")
        # Print an error message if the file does not exist.

if __name__ == '__main__':
    # Check if the script is being run directly.
    create_gloss_set()
    # Call the function to execute the script's main logic.
