#!/bin/bash
set -e 

# --- Configuration - ADJUST THESE PATHS IF DIFFERENT FROM DEFAULTS ---
# Directory where your pre-processed data (custom_char_dict.txt, rec_gt_train.txt, rec_gt_eval.txt, line_images/) is stored
export PROCESSED_DATA_DIR="/home/jupyter/PaddleOCR_Training/ocr_output"
export CHAR_DICT_FILE_PATH="${PROCESSED_DATA_DIR}/custom_char_dict.txt"
export TRAIN_LABEL_FILE_PATH="${PROCESSED_DATA_DIR}/rec_gt_train.txt"
export EVAL_LABEL_FILE_PATH="${PROCESSED_DATA_DIR}/rec_gt_eval.txt"

# Path where you have cloned the PaddleOCR repository
export PADDLE_OCR_REPO_PATH="/home/jupyter/PaddleOCR" 

# Path to your YAML configuration file (ensure this is the one updated in the previous step)
# The YAML config should already have absolute paths to char_dict, train/eval labels, and pretrained_model
export PADDLE_OCR_CONFIG_FILE="${PADDLE_OCR_REPO_PATH}/configs/rec/my_config_rec_ppocrv4_finetune.yml"

# Path to the PARENT directory of the downloaded PRETRAINED PP-OCRv4 English RECOGNITION model
# The YAML file should have the specific path to the .pdparams file or model directory.
# This variable is mainly for user reference here.
export PRETRAINED_MODEL_PARENT_DIR="/home/jupyter/PaddleOCR_Training/pretrained_models" 

echo "--- Script Configuration ---"
echo "Processed Data Directory: ${PROCESSED_DATA_DIR}"
echo "Character Dictionary: ${CHAR_DICT_FILE_PATH}"
echo "Training Labels: ${TRAIN_LABEL_FILE_PATH}"
echo "Evaluation Labels: ${EVAL_LABEL_FILE_PATH}"
echo "PaddleOCR Repo Path: ${PADDLE_OCR_REPO_PATH}"
echo "PaddleOCR Config File: ${PADDLE_OCR_CONFIG_FILE}"
echo "Pretrained Model Parent Dir: ${PRETRAINED_MODEL_PARENT_DIR}"
echo "----------------------------"

# --- Sanity Checks ---
echo "Performing Sanity Checks..."
if [ ! -d "$PROCESSED_DATA_DIR" ]; then echo "ERROR: Processed data directory not found: $PROCESSED_DATA_DIR"; exit 1; fi
if [ ! -f "$CHAR_DICT_FILE_PATH" ]; then echo "ERROR: Character dictionary not found: $CHAR_DICT_FILE_PATH"; exit 1; fi
if [ ! -f "$TRAIN_LABEL_FILE_PATH" ]; then echo "ERROR: Training label file not found: $TRAIN_LABEL_FILE_PATH"; exit 1; fi
if [ ! -f "$EVAL_LABEL_FILE_PATH" ]; then echo "ERROR: Evaluation label file not found: $EVAL_LABEL_FILE_PATH"; exit 1; fi
if [ ! -d "$PADDLE_OCR_REPO_PATH" ]; then echo "ERROR: PaddleOCR repository path not found: $PADDLE_OCR_REPO_PATH"; exit 1; fi
if [ ! -f "$PADDLE_OCR_CONFIG_FILE" ]; then echo "ERROR: PaddleOCR config file not found: $PADDLE_OCR_CONFIG_FILE"; exit 1; fi
# You might also want to check if the pretrained model path specified IN THE YAML exists, but that's harder from here.
echo "Sanity checks passed."
echo "----------------------------------------------------------------------"

# --- Step 1: Fine-tuning the PaddleOCR Recognition Model ---
# (Was Step 4 in the previous script)
echo "STEP 1: Fine-tuning the PaddleOCR Recognition Model"
echo "----------------------------------------------------------------------"
echo "Your YAML config file ($PADDLE_OCR_CONFIG_FILE) should have all paths correctly set to:"
echo "  - Global.character_dict_path: ${CHAR_DICT_FILE_PATH}"
echo "  - Global.max_text_length: 128 (or as set in YAML)"
echo "  - Global.pretrained_model: (e.g. ${PRETRAINED_MODEL_PARENT_DIR}/en_PP-OCRv4_rec_train/best_accuracy or the actual path in YAML)"
echo "  - Train.dataset.label_file_list: [\"${TRAIN_LABEL_FILE_PATH}\"]"
echo "  - Eval.dataset.label_file_list: [\"${EVAL_LABEL_FILE_PATH}\"]"
echo ""
echo "IMPORTANT: Ensure the 'pretrained_model' path within '$PADDLE_OCR_CONFIG_FILE' is correct and points to your downloaded model."
echo ""
echo "Press Enter to continue with training, or Ctrl+C to review/edit the YAML config first."
read

cd "$PADDLE_OCR_REPO_PATH"

echo "Starting training..."
# Ensure you are in the PaddleOCR virtual environment (e.g., ocr_env)
# The training command assumes your YAML is correctly configured.
python tools/train.py -c "$PADDLE_OCR_CONFIG_FILE"

# Note: The actual save_model_dir is read from the YAML during the export step.
# This is just an informational message.
SAVE_MODEL_DIR_FROM_YAML_INFO=$(grep 'save_model_dir:' "$PADDLE_OCR_CONFIG_FILE" | awk '{print $2}')
echo "Training finished. Checkpoint models should be in ${PADDLE_OCR_REPO_PATH}/${SAVE_MODEL_DIR_FROM_YAML_INFO}" 
echo "----------------------------------------------------------------------"

# --- Step 2: Exporting the fine-tuned model for inference ---
# (Was Step 5 in the previous script)
echo "STEP 2: Exporting the fine-tuned model for inference"
echo "----------------------------------------------------------------------"
# Determine save_model_dir and save_inference_dir from YAML for accuracy
# These paths are relative to the PADDLE_OCR_REPO_PATH
BEST_MODEL_PATH_FROM_YAML=$(grep 'save_model_dir:' "$PADDLE_OCR_CONFIG_FILE" | awk '{print $2}')/best_accuracy 
SAVE_INFERENCE_DIR_FROM_YAML=$(grep 'save_inference_dir:' "$PADDLE_OCR_CONFIG_FILE" | awk '{print $2}')

echo "Exporting model from checkpoint: ${PADDLE_OCR_REPO_PATH}/${BEST_MODEL_PATH_FROM_YAML}"
echo "Inference model will be saved to directory (relative to PaddleOCR repo): ${SAVE_INFERENCE_DIR_FROM_YAML}"

python tools/export_model.py \
    -c "$PADDLE_OCR_CONFIG_FILE" \
    -o Global.pretrained_model="${BEST_MODEL_PATH_FROM_YAML}" \
    -o Global.save_inference_dir="${SAVE_INFERENCE_DIR_FROM_YAML}"

echo "Inference model exported to ${PADDLE_OCR_REPO_PATH}/${SAVE_INFERENCE_DIR_FROM_YAML}"
cd - > /dev/null # Go back to previous directory silently
echo "----------------------------------------------------------------------"
echo "Training and Export Pipeline Finished!"
echo "----------------------------------------------------------------------"
