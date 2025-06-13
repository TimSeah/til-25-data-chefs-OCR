import os
import pandas as pd

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

def create_labels_csv(image_folder: str, output_csv_path: str):
    image_files = []
    text_contents = []
    found_pairs = 0
    missing_text_for_images = 0
    print(f"Scanning folder: {image_folder}")
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder not found or is not a directory: {image_folder}")
        return
    for filename in os.listdir(image_folder):
        file_basename, file_extension = os.path.splitext(filename)
        if file_extension.lower() in IMAGE_EXTENSIONS:
            current_image_path = os.path.join(image_folder, filename)
            text_to_add = None
            text_filename_pattern1 = f"{file_basename}_text.txt"
            text_filepath_pattern1 = os.path.join(image_folder, text_filename_pattern1)
            if os.path.exists(text_filepath_pattern1):
                try:
                    with open(text_filepath_pattern1, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content:
                        text_to_add = content
                    else:
                        print(f"Info: Text file '{text_filename_pattern1}' for image '{filename}' is empty.")
                except Exception as e:
                    print(f"Warning: Error reading text file '{text_filename_pattern1}' for image '{filename}': {e}")
            if text_to_add is None:
                text_filename_pattern2 = f"{file_basename}.txt"
                text_filepath_pattern2 = os.path.join(image_folder, text_filename_pattern2)
                if os.path.exists(text_filepath_pattern2):
                    try:
                        with open(text_filepath_pattern2, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        if content:
                            text_to_add = content
                        else:
                            print(f"Info: Text file '{text_filename_pattern2}' for image '{filename}' is empty.")
                    except Exception as e:
                        print(f"Warning: Error reading text file '{text_filename_pattern2}' for image '{filename}': {e}")
            if text_to_add:
                image_files.append(current_image_path)
                text_contents.append(text_to_add)
                found_pairs += 1
            else:
                missing_text_for_images += 1
                if not os.path.exists(text_filepath_pattern1) and \
                   not os.path.exists(os.path.join(image_folder, f"{file_basename}.txt")):
                    print(f"Info: No corresponding text file ('{text_filename_pattern1}' or '{file_basename}.txt') found for image '{filename}'.")
    if not image_files:
        print("No image-text pairs found. Please check file naming conventions and folder content.")
        return
    df = pd.DataFrame({'image_path': image_files, 'text': text_contents})
    try:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully created '{output_csv_path}' with {found_pairs} image-text pairs.")
        if missing_text_for_images > 0:
            print(f"Could not find or process valid text for {missing_text_for_images} images.")
    except Exception as e:
        print(f"Error: Could not write CSV file to '{output_csv_path}': {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Input image folder (assuming this path is still correct)
    image_folder_path = "/home/jupyter/advanced/ocr/" 
    
    # UPDATED: Output directory for labels.csv
    writable_output_directory = "/home/jupyter/PaddleOCR_Training/ocr_output/" 
    output_csv_file = os.path.join(writable_output_directory, "labels.csv")

    # --- Run the script ---
    create_labels_csv(image_folder_path, output_csv_file)