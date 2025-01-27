import os
import shutil
import random

def copy_random_file_from_subfolders(src_dir, dest_dir):
    """
    Copies one random .png file from each subfolder (0-42) in the source directory 
    and saves it in the destination directory with a renamed filename.
    
    Args:
        src_dir (str): Path to the source directory containing subfolders 0-42.
        dest_dir (str): Path to the destination directory where files will be saved.
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate through subfolders named 0 to 42
    for folder_name in range(43):  # Subfolders 0 to 42
        folder_path = os.path.join(src_dir, str(folder_name))
        
        if not os.path.exists(folder_path):
            print(f"Subfolder {folder_name} does not exist. Skipping...")
            continue
        
        # List all .png files in the current subfolder
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
        if not png_files:
            print(f"No .png files found in subfolder {folder_name}. Skipping...")
            continue

        # Randomly select one .png file
        random_file = random.choice(png_files)

        # Copy the selected file to the destination directory
        src_file_path = os.path.join(folder_path, random_file)
        dest_file_name = f"class_{folder_name}.png"
        dest_file_path = os.path.join(dest_dir, dest_file_name)
        
        shutil.copy(src_file_path, dest_file_path)
        print(f"Copied {random_file} from subfolder {folder_name} to {dest_file_name}.")

# Source and destination directory paths
source_directory = "DatasetUnzipped/test"  # Replace with the actual path
destination_directory = "TestEachClass"  # Replace with the actual path

# Execute the function
copy_random_file_from_subfolders(source_directory, destination_directory)
