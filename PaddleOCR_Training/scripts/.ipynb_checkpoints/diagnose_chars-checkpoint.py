import csv
import sys

def check_csv_chars(csv_file_path):
    """
    Reads a CSV file and prints all unique characters found in the 'text' column.
    """
    unique_chars = set()
    line_count = 0
    text_column_idx = -1

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            try:
                header = next(reader)
                line_count += 1
                if 'text' in header:
                    text_column_idx = header.index('text')
                else:
                    print("ERROR: 'text' column not found in CSV header.", file=sys.stderr)
                    return
            except StopIteration:
                print("ERROR: CSV file is empty or has no header.", file=sys.stderr)
                return
            
            print(f"Found 'text' column at index: {text_column_idx}", file=sys.stderr)
            print("Processing lines (Ctrl+C to stop early if it takes too long)...", file=sys.stderr)

            for i, row in enumerate(reader):
                line_count += 1
                if text_column_idx < 0 or text_column_idx >= len(row):
                    if i < 5: # Print error for first few occurrences
                        print(f"Warning: Row {line_count} does not have a valid entry for 'text' column (index {text_column_idx}, row length {len(row)}). Skipping.", file=sys.stderr)
                    continue
                
                text_line = row[text_column_idx]
                unique_chars.update(text_line)

                if i % 50000 == 0 and i > 0:
                    print(f"Processed {line_count} lines. Current unique chars: {len(unique_chars)}", file=sys.stderr)

    except FileNotFoundError:
        print(f"ERROR: File not found: {csv_file_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return

    print(f"\n--- Diagnostic Complete ---", file=sys.stderr)
    print(f"Total lines processed from CSV: {line_count-1}", file=sys.stderr) # -1 for header
    print(f"Total unique characters found in '{csv_file_path}': {len(unique_chars)}", file=sys.stderr)
    
    print("\n--- Unique Characters (one per line) ---")
    for char in sorted(list(unique_chars)):
        print(char)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_chars.py <path_to_line_labels.csv>", file=sys.stderr)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    check_csv_chars(csv_path)