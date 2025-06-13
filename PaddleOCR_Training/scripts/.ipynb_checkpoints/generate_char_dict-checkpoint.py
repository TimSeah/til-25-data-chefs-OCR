import pandas as pd
import argparse
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_dictionary(csv_file_path, output_dict_path):
    try:
        logging.info(f"Attempting to read CSV: {csv_file_path}")
        # Read only the 'text' column
        try:
            # Specify dtype as str during read to prevent misinterpretation
            df = pd.read_csv(csv_file_path, usecols=['text'], dtype={'text': str})
        except ValueError: # Fallback if 'text' column is the only column or other read_csv issue
            logging.warning("Failed to read only 'text' column, trying to read full CSV.")
            df = pd.read_csv(csv_file_path, dtype={'text': str}) # Ensure text column is treated as string
            if 'text' not in df.columns:
                 logging.error(f"'text' column not found in {csv_file_path}")
                 return False
        
        logging.info(f"Successfully read CSV. Number of rows: {len(df)}")
        
        # Ensure 'text' column is not NaN and is string type
        df.dropna(subset=['text'], inplace=True)
        df['text'] = df['text'].astype(str)
        logging.info(f"Number of rows after dropping NaN in 'text': {len(df)}")

        if df.empty:
            logging.warning("No text data found in CSV after initial processing.")
            with open(output_dict_path, 'w', encoding='utf-8') as f:
                pass 
            logging.info(f"Empty character dictionary generated at: {output_dict_path} due to no input text.")
            return True

        unique_chars = set()
        
        # Iterate row by row to build the set of unique characters
        # This is more memory-efficient than creating one giant string
        logging.info("Processing text lines to find unique characters...")
        for text_line in tqdm(df['text'], desc="Finding unique characters"):
            unique_chars.update(text_line) # Add all characters from the current line to the set
        
        if not unique_chars:
            logging.warning("No unique characters found after processing all text lines.")
            with open(output_dict_path, 'w', encoding='utf-8') as f:
                pass
            logging.info(f"Empty character dictionary generated at: {output_dict_path} due to no input text.")
            return True

        # Sort the characters before writing
        sorted_unique_chars = sorted(list(unique_chars))
        
        with open(output_dict_path, 'w', encoding='utf-8') as f:
            for char in sorted_unique_chars:
                f.write(char + "\n")
        
        logging.info(f"Character dictionary generated at: {output_dict_path}")
        logging.info(f"Total unique characters written to dictionary: {len(sorted_unique_chars)}")
        
        # Provide a more detailed warning if character count is low
        if len(sorted_unique_chars) < 80 and len(sorted_unique_chars) > 0: # Increased threshold
             logging.warning(
                f"The number of unique characters ({len(sorted_unique_chars)}) still seems low for a diverse dataset. "
                f"Please verify the content of '{output_dict_path}' and the source text data in '{csv_file_path}'. "
                f"Common missing items could be numbers (0-9), all uppercase (A-Z), "
                f"all lowercase (a-z), or common punctuation (e.g., !?\"#$%&'()*+,-./:;<=>@[\]^_`{{|}}~)."
             )
        elif len(sorted_unique_chars) == 0:
            logging.error("CRITICAL: The generated character dictionary is empty but there was input text!")

        return True
        
    except FileNotFoundError:
        logging.error(f"CSV file not found at {csv_file_path}")
        return False
    except Exception as e:
        logging.error(f"An error occurred in generate_dictionary: {e}", exc_info=True) # Enable full traceback for this
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a character dictionary from a CSV file.")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("output_dict_file", help="Path to save the generated character dictionary.")
    args = parser.parse_args()
    
    output_dir = os.path.dirname(os.path.abspath(args.output_dict_file))
    if not output_dir: # Handle case where output_dict_file might be just a filename
        output_dir = "." # Default to current directory
    os.makedirs(output_dir, exist_ok=True)

    if not generate_dictionary(args.csv_file, args.output_dict_file):
        print(f"Failed to generate character dictionary. Check logs for details.")
        # exit(1) # Optionally exit with error code
    else:
        print(f"generate_char_dict.py completed. Dictionary at: {args.output_dict_file}")