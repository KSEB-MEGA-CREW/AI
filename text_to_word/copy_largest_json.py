
import os # Import the os module for interacting with the operating system
import shutil # Import the shutil module for high-level file operations
import json # Import the json module for working with JSON data

def copy_largest_json(): # Define the main function
    """
    Finds the largest JSON file in each subdirectory of 'video_to_json'
    and copies it to 'selected_json' with a new name based on the subdirectory.
    """
    # Define the source directory path
    source_dir = r'C:\Users\Hyuk\Documents\대학\부트켐프\메인 프로젝트\AI\video_to_json'
    # Define the destination directory path
    dest_dir = r'C:\Users\Hyuk\Documents\대학\부트켐프\메인 프로젝트\AI\selected_json'

    # Create the destination directory if it doesn't already exist
    if not os.path.exists(dest_dir):
        # Make the directory, including any necessary parent directories
        os.makedirs(dest_dir)

    # Iterate over each item in the source directory
    for subdir in os.listdir(source_dir):
        # Create the full path to the subdirectory
        subdir_path = os.path.join(source_dir, subdir)
        # Check if the item is a directory
        if os.path.isdir(subdir_path):
            # Get a list of all files in the subdirectory that end with .json
            json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
            # If there are no JSON files, skip to the next subdirectory
            if not json_files:
                # Continue to the next iteration of the loop
                continue

            # Initialize a variable to store the path of the largest file
            largest_file = ''
            # Initialize a variable to store the size of the largest file
            largest_size = -1

            # Iterate over each JSON file found in the subdirectory
            for json_file in json_files:
                # Create the full path to the JSON file
                file_path = os.path.join(subdir_path, json_file)
                # Start a try block to handle potential errors
                try:
                    # Get the size of the file
                    file_size = os.path.getsize(file_path)
                    # Check if the current file is larger than the largest one found so far
                    if file_size > largest_size:
                        # If it is, update the largest size
                        largest_size = file_size
                        # And update the path of the largest file
                        largest_file = file_path
                # Catch OSError, which can occur if the file can't be accessed
                except OSError:
                    # Ignore files that can't be accessed and do nothing
                    pass

            # Check if a largest file was found in the subdirectory
            if largest_file:
                # Create the new filename by replacing spaces with underscores in the subdirectory name
                new_filename = subdir.replace(' ', '_') + '.json'
                # Create the full destination path including the new filename
                dest_path = os.path.join(dest_dir, new_filename)
                # Print a message indicating which file is being copied and to where
                print(f"Copying {largest_file} to {dest_path}")
                # Copy the largest file to the destination directory with the new name
                shutil.copy(largest_file, dest_path)

# Check if the script is being run directly
if __name__ == '__main__':
    # Call the main function
    copy_largest_json()
