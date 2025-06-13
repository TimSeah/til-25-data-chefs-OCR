import pandas as pd
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_max_text_length(csv_file_path):
    try:
        # Read only the 'text' column
        try:
            df = pd.read_csv(csv_file_path, usecols=['text'])
        except ValueError:
            df = pd.read_csv(csv_file_path)
            if 'text' not in df.columns:
                logging.error(f"'text' column not found in {csv_file_path}")
                return -1 # Indicate error

        df.dropna(subset=['text'], inplace=True)
        df['text'] = df['text'].astype(str)
        
        if df.empty or df['text'].empty:
            logging.warning("No text data found in CSV to calculate max length.")
            return 0 

        # Calculate max length efficiently
        max_len = df['text'].map(len).max()
        
        # print(f"Maximum text length found: {max_len}") # Standard output for shell script to capture
        return int(max_len) # Ensure it's an int
        
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_file_path}")
        return -1
    except Exception as e:
        logging.error(f"An error occurred in get_max_text_length: {e}", exc_info=False)
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the maximum text length from a CSV file.")
    parser.add_argument("csv_file", help="Path to the input CSV file (e.g., ../ocr_output/line_labels.csv)")
    args = parser.parse_args()

    max_length = get_max_text_length(args.csv_file)
    if max_length >= 0: # Successfully calculated (0 or more)
        print(max_length) # Print only the number for capture by shell script
    else:
        # Optionally print an error to stderr if needed, or let logging handle it
        # print("Error calculating max length.", file=sys.stderr)
        exit(1) # Exit with error code if calculation failed