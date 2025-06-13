# src/data_preprocess.py

import os
import re
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError # Added Pillow

# Setup basic logging - INFO level
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Determine parser: prefer lxml's XML parser, then lxml's HTML, then html.parser
BS_PARSER_TYPE = None
try:
    import lxml
    BS_PARSER_TYPE = 'xml'
    logging.info("BeautifulSoup will use 'lxml' (XML mode).")
except ImportError:
    BS_PARSER_TYPE = 'html.parser'
    logging.info("BeautifulSoup will use 'html.parser'. XMLParsedAsHTMLWarning may occur for HOCR files.")
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning, module='bs4')

def extract_text_from_hocr_element(element):
    words = element.find_all('span', class_='ocrx_word')
    if words:
        line_text = ' '.join([word.get_text(strip=True) for word in words])
    else:
        line_text = element.get_text(strip=True)
    line_text = re.sub(r'\s+', ' ', line_text).strip()
    return line_text

def parse_bbox_from_title(title_string):
    if not title_string:
        return None
    match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title_string)
    if match:
        return tuple(map(int, match.groups())) # (x0, y0, x1, y1)
    return None

def process_single_hocr(hocr_file_path, full_image_path, cropped_images_dir):
    """
    Processes a single HOCR file to extract line-level text and corresponding cropped images.
    """
    line_data = []
    try:
        with open(hocr_file_path, 'r', encoding='utf-8') as f:
            hocr_content = f.read()
        
        soup = BeautifulSoup(hocr_content, BS_PARSER_TYPE)
        potential_line_elements = soup.find_all('span', class_=['ocr_line', 'ocr_header'])
        
        if not potential_line_elements:
            return line_data

        # Open the full image once
        try:
            original_pil_image = Image.open(full_image_path)
        except FileNotFoundError:
            logging.error(f"Full image not found: {full_image_path} for HOCR {hocr_file_path}")
            return line_data
        except UnidentifiedImageError:
            logging.error(f"Cannot identify image file (corrupted or unsupported format): {full_image_path}")
            return line_data
        except Exception as img_e:
            logging.error(f"Error opening image {full_image_path}: {img_e}")
            return line_data

        full_image_basename = os.path.splitext(os.path.basename(full_image_path))[0]

        for i, element in enumerate(potential_line_elements):
            element_text = extract_text_from_hocr_element(element)
            if not element_text:
                continue

            title = element.get('title', '')
            bbox = parse_bbox_from_title(title)

            if not bbox:
                logging.warning(f"Could not parse bbox for line in {hocr_file_path}, element ID: {element.get('id', 'N/A')}. Skipping line.")
                continue
            
            # Ensure coordinates are valid (x0 < x1, y0 < y1)
            if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                logging.warning(f"Invalid bbox coordinates {bbox} in {hocr_file_path} for element ID: {element.get('id', 'N/A')}. Skipping line.")
                continue

            try:
                # Crop the line image from the full image
                # HOCR bbox is (x0, y0, x1, y1) which matches Pillow's crop ((left, upper, right, lower))
                line_image_cropped = original_pil_image.crop(bbox)
                
                # Create a unique filename for the cropped image
                # Format: <original_image_basename>_line_<index_in_hocr>_<x0>_<y0>_<x1>_<y1>.png
                cropped_image_filename = f"{full_image_basename}_line_{i}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png"
                cropped_image_save_path = os.path.join(cropped_images_dir, cropped_image_filename)
                
                line_image_cropped.save(cropped_image_save_path, "PNG") # Save as PNG

                line_data.append({'image_path': cropped_image_save_path, 'text': element_text})

            except Exception as e:
                logging.error(f"Error cropping/saving line from {full_image_path} (element ID: {element.get('id', 'N/A')}) with bbox {bbox}: {e}", exc_info=False)
        
        # Close the full image if it was opened
        if 'original_pil_image' in locals() and hasattr(original_pil_image, 'close'):
             original_pil_image.close()

    except FileNotFoundError:
        logging.error(f"HOCR file not found: {hocr_file_path}")
    except Exception as e:
        logging.error(f"Error processing HOCR file {hocr_file_path}: {e}", exc_info=False)
        
    return line_data

def create_line_labels_csv(image_dir, hocr_dir, output_csv_path, num_workers=None):
    all_line_data = []
    image_files = []
    
    # Define and create the directory for cropped line images
    # It will be a subdirectory in the same directory as the output_csv_path
    csv_dir = os.path.dirname(os.path.abspath(output_csv_path))
    cropped_images_output_dir = os.path.join(csv_dir, "line_images") # e.g., ocr_output/line_images/
    os.makedirs(cropped_images_output_dir, exist_ok=True)
    logging.info(f"Cropped line images will be saved to: {cropped_images_output_dir}")

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(os.path.join(root, file))
    image_files.sort()

    if not image_files:
        logging.error(f"No image files found in {image_dir}")
        return False

    tasks = []
    for image_file_path in image_files:
        base_name = os.path.splitext(os.path.basename(image_file_path))[0]
        hocr_file_name = base_name + ".hocr"
        hocr_file_path = os.path.join(hocr_dir, hocr_file_name)

        if os.path.exists(hocr_file_path):
            tasks.append((hocr_file_path, image_file_path, cropped_images_output_dir))
        else:
            hocr_file_name_alt = os.path.basename(image_file_path) + ".hocr" # Alternative: image.png.hocr
            hocr_file_path_alt = os.path.join(hocr_dir, hocr_file_name_alt)
            if os.path.exists(hocr_file_path_alt):
                 tasks.append((hocr_file_path_alt, image_file_path, cropped_images_output_dir))
            else:
                logging.warning(f"HOCR file not found for image {image_file_path} (tried {hocr_file_name}, {hocr_file_name_alt})")

    if not tasks:
        logging.error("No HOCR files found to process with associated images.")
        return False

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_hocr, hfp, ifp, ciod) for hfp, ifp, ciod in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing HOCR files"):
            try:
                result = future.result()
                if result: 
                    all_line_data.extend(result)
            except Exception as e:
                logging.error(f"A HOCR processing task generated an exception: {e}", exc_info=False)

    if not all_line_data:
        logging.error("No line data extracted from any HOCR file after processing.")
        return False

    df = pd.DataFrame(all_line_data)
    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        logging.info(f"Successfully created line-level labels CSV: {output_csv_path} with {len(df)} entries.")
    except Exception as e:
        logging.error(f"Failed to write CSV to {output_csv_path}: {e}")
        return False
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create line-level labels and cropped images from HOCR files for OCR training.")
    parser.add_argument("image_directory", help="Directory containing the original full-page image files.")
    parser.add_argument("hocr_directory", help="Directory containing the HOCR files.")
    parser.add_argument("output_csv", help="Path to save the output CSV file (e.g., /path/to/ocr_output/line_labels.csv).")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes. Defaults to CPU count if None.")
    args = parser.parse_args()

    # Ensure the directory for the output CSV exists
    output_csv_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if not output_csv_dir: # Handle cases where output_csv might be just a filename
        output_csv_dir = "."
    os.makedirs(output_csv_dir, exist_ok=True)

    if create_line_labels_csv(args.image_directory, args.hocr_directory, args.output_csv, args.workers):
        print(f"data_preprocess.py completed. Output CSV: {args.output_csv}")
    else:
        print("data_preprocess.py failed.")