import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import logging

# Modified by Copilot for lolkabash
# Current User: lolkabash
# Current Date (UTC): 2025-05-25 00:34:45 (as per user context)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def convert_labels(csv_file_path, output_dir, train_ratio=0.9, char_dict_path=None, max_text_length=None):
    unknown_chars_found = set() 

    try:
        try:
            df = pd.read_csv(csv_file_path, usecols=['image_path', 'text'])
        except ValueError:
            df = pd.read_csv(csv_file_path)
            if 'image_path' not in df.columns or 'text' not in df.columns:
                logging.error(f"'image_path' or 'text' column not found in {csv_file_path}")
                return False

        df['text'] = df['text'].astype(str)
        df['image_path'] = df['image_path'].astype(str)
        
        df.dropna(subset=['image_path', 'text'], inplace=True)
        df = df[df['image_path'].str.strip().astype(bool)]
        df = df[df['text'].str.strip().astype(bool)]

        initial_lines = len(df)
        logging.info(f"Initial number of lines after loading and basic cleaning: {initial_lines}")

        if df.empty:
            logging.error("No data to process after basic cleaning.")
            return False
            
        valid_chars = None
        if char_dict_path:
            try:
                with open(char_dict_path, 'r', encoding='utf-8') as f:
                    # Corrected dictionary loading:
                    # Read each line, rstrip() the newline, and if it's not empty, add it.
                    # This preserves lines that are just a space.
                    raw_lines = f.readlines()
                    valid_chars = set()
                    for line_content in raw_lines:
                        char = line_content.rstrip('\n\r') # Remove only newlines
                        if char: # Add if the result (e.g. " " or "a") is not empty
                            valid_chars.add(char)
                        elif line_content == '\n' or line_content == '\r\n': # Handle empty lines if any, don't add as char
                            pass # Explicitly do nothing for truly empty lines
                        # If a line was " \n", char becomes " ". " " is not empty, so it's added.
                        # If a line was "\n", char becomes "". "" is empty, so it's skipped.

                logging.info(f"Loaded {len(valid_chars)} characters from dictionary: {char_dict_path}")
                if not valid_chars: # Should not happen if dict has content
                    logging.warning(f"Character dictionary '{char_dict_path}' appears empty after loading.")
            except FileNotFoundError:
                logging.warning(f"Character dictionary '{char_dict_path}' not found. No character validation.")
            except Exception as e:
                logging.warning(f"Error reading char dict '{char_dict_path}': {e}. No char validation.", exc_info=False)
        
        if valid_chars:
            original_count_before_char_filter = len(df)
            lines_to_keep_char = []
            num_filtered_out_char = 0
            
            for index, row in df.iterrows():
                text_line = row['text']
                is_valid_line = True
                for char_in_text in text_line:
                    if char_in_text not in valid_chars:
                        is_valid_line = False
                        unknown_chars_found.add(char_in_text)
                        break # No need to check further chars in this line
                
                if is_valid_line:
                    lines_to_keep_char.append(index)
                else:
                    num_filtered_out_char +=1

            df = df.loc[lines_to_keep_char]
            
            if num_filtered_out_char > 0:
                logging.info(f"Filtered out {num_filtered_out_char} lines (out of {original_count_before_char_filter}) due to characters not in dictionary.")
            
            if len(df) == 0 and original_count_before_char_filter > 0 :
                logging.error(f"Error: No lines remaining after filtering with char dict. Check dict and data compatibility.")
        else:
            logging.info("No character dictionary provided/loaded, so no character-based filtering applied.")

        if max_text_length is not None and max_text_length > 0:
            original_count_before_len_filter = len(df)
            if original_count_before_len_filter > 0:
                df = df[df['text'].str.len() <= max_text_length]
                num_filtered_out_len = original_count_before_len_filter - len(df)
                if num_filtered_out_len > 0:
                    logging.info(f"Filtered out {num_filtered_out_len} lines (out of {original_count_before_len_filter}) due to text length exceeding {max_text_length} characters.")
                if len(df) == 0 and original_count_before_len_filter > 0 and num_filtered_out_len > 0: # check if filtering actually removed all
                    logging.error(f"Error: No lines remaining after filtering by max_text_length of {max_text_length}.")
            elif original_count_before_len_filter == 0: # If df was already empty
                 logging.info("No lines to filter by length as dataframe is already empty.")
            else: # If df was not empty, but no lines were filtered by length
                logging.info(f"No lines filtered out due to text length exceeding {max_text_length} characters.")

        if unknown_chars_found:
            unknown_chars_file_path = os.path.join(output_dir, "unknown_characters.txt")
            try:
                with open(unknown_chars_file_path, 'w', encoding='utf-8') as f_unknown:
                    for char_val in sorted(list(unknown_chars_found)):
                        f_unknown.write(f"{char_val}\n")
                logging.info(f"Found {len(unknown_chars_found)} unique unknown characters. Saved to: {unknown_chars_file_path}")
            except Exception as e:
                logging.error(f"Could not write unknown characters file: {e}")
        elif valid_chars: # Only log "no unknown" if a dictionary was actually used for checking
            logging.info("No unknown characters found in the dataset relative to the provided dictionary.")


        if df.empty:
            logging.error("Error: No data to process for splitting into train/eval sets after all filtering.")
            return False

        train_df, eval_df = train_test_split(df, train_size=train_ratio, random_state=42, shuffle=True)
        
        os.makedirs(output_dir, exist_ok=True)
        train_label_path = os.path.join(output_dir, "rec_gt_train.txt")
        eval_label_path = os.path.join(output_dir, "rec_gt_eval.txt")

        with open(train_label_path, 'w', encoding='utf-8') as f_train:
            for _, row in train_df.iterrows():
                f_train.write(f"{row['image_path']}\t{row['text']}\n")
        
        with open(eval_label_path, 'w', encoding='utf-8') as f_eval:
            for _, row in eval_df.iterrows():
                f_eval.write(f"{row['image_path']}\t{row['text']}\n")

        logging.info(f"PaddleOCR training labels created:")
        logging.info(f"  Train: {train_label_path} ({len(train_df)} lines)")
        logging.info(f"  Eval: {eval_label_path} ({len(eval_df)} lines)")
        return True

    except FileNotFoundError:
        logging.error(f"CSV file not found at {csv_file_path}")
        return False
    except Exception as e:
        logging.error(f"An error occurred during label conversion: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV labels to PaddleOCR recognition format.")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("output_dir", help="Directory to save rec_gt_train.txt and rec_gt_eval.txt.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio for training (default: 0.9)")
    parser.add_argument("--char_dict", type=str, default=None, help="Path to character dictionary for filtering.")
    parser.add_argument("--max_text_length", type=int, default=None, help="Maximum allowed text length for filtering (default: None, no filtering).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not convert_labels(args.csv_file, args.output_dir, args.train_ratio, args.char_dict, args.max_text_length):
        logging.error("Label conversion process failed. Please check logs above.")
