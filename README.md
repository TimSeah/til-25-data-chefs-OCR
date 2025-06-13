# TIL-25 Data Chefs - OCR Challenge

**Hackathon:** TIL-25 Hackathon
**Team:** Data Chefs
**Author:** lolkabash

## 📖 Description

This repository contains the solution for the OCR (Optical Character Recognition) challenge as part of the TIL-25 Hackathon. The primary goal was to train an effective OCR model.

*(You can add more specific details about the challenge problem here if you like.)*

## 💻 Technologies Used

*   **Python:** Core programming language for model development and scripting.
*   **Jupyter Notebook:** Used for experimentation, data exploration, and model training iterations.
*   **Shell Scripts:** For automation of tasks like data preprocessing, training initiation, etc.
*   **PaddleOCR:** For the initial pretrained model and finetuning.
*   **OpenCV, NumPy, Pandas, Matplotlib, PyTorch/TensorFlow**

## ⚙️ Working Process & Solution

This section outlines the general steps taken to address the OCR challenge.

### 1. Data Collection & Preparation
*   **Dataset Used:** (Describe the dataset(s) used, e.g., public datasets, custom collected data. Mention size, type of images, etc.)
*   **Preprocessing:** (Detail the steps taken to clean and prepare the images for the OCR model, e.g., resizing, noise reduction, binarization, augmentation.)
*   **Labeling:** (If custom data was used, how was it labeled? E.g., tools used like LabelImg, or programmatic approaches.)

### 2. Model Selection & Architecture
*   **Model Choice:** (Explain why a particular OCR model or architecture was chosen. E.g., CRNN, ViT-based models, specific pre-trained models.)
*   **Architecture Details:** (Briefly describe the model architecture if it was custom or significantly modified.)
*   **Pre-trained Models:** (Specify if any pre-trained weights were used as a starting point, e.g., from PaddleOCR model zoo.)

### 3. Training Process
*   **Environment Setup:** (Briefly mention the environment, e.g., local machine specs, cloud VM, specific Python/library versions.)
*   **Training Configuration:** (Key hyperparameters, loss functions, optimizers, batch size, number of epochs.)
*   **Fine-tuning:** (If a pre-trained model was used, describe the fine-tuning strategy.)
*   **Challenges Faced:** (Any significant challenges during training and how they were overcome.)

### 4. Evaluation
*   **Metrics Used:** (How was the model performance measured? E.g., Character Error Rate (CER), Word Error Rate (WER), accuracy, precision, recall.)
*   **Validation Strategy:** (How was the model validated during training? E.g., validation set, cross-validation.)
*   **Test Set Performance:** (Results on the final test set.)

### 5. Results & Key Findings
*   **Final Model Performance:** (Summarize the best results achieved.)
*   **Insights:** (Any interesting insights gained from the process or results.)
*   **Visualizations:** (Consider linking to or embedding examples of OCR output if possible.)

## 🚀 Setup and Usage

### Prerequisites
*   Python version 3.10+
*   Git & Git LFS
*   CUDA version 12.2.1+

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/lolkabash/til-25-data-chefs-OCR.git
    cd til-25-data-chefs-OCR
    ```
2.  (If Git LFS was used for model files, etc.)
    ```bash
    git lfs pull
    ```
3.  Install dependencies:
    ```bash
    # conda env create -f environment.yml
    # conda activate your_env_name
    ```

### Running the Code
*   **Data Preparation:**
    ```bash
    # e.g., python scripts/prepare_data.py
    ```
*   **Training:**
    *(Explain how to run the training scripts or notebooks.)*
    ```bash
    # e.g., python train_ocr.py --config configs/my_config.yaml
    # or jupyter notebook PaddleOCR_Training/CreateLabel.ipynb (based on previous interactions)
    ```
*   **Inference/Prediction:**
    *(Explain how to use the trained model for predictions.)*
    ```bash
    # e.g., python predict.py --image_path path/to/image.png --model_path path/to/model
    ```

## 📁 File Structure
```
til-25-data-chefs-OCR/
├── PaddleOCR_Training/         # Main training scripts, notebooks, and model files (as per previous interactions)
│   ├── CreateLabel.ipynb
│   └── pretrained_models/
│       └── en_PP-OCRv4_rec_train/
│           ├── best_accuracy.pdparams
│           └── best_accuracy.pdopt
├── configs/                    # Configuration files for training
├── data/                       # Placeholder for datasets (ensure .gitignore if data is large and not in LFS)
├── notebooks/                  # Jupyter notebooks for exploration, analysis
├── scripts/                    # Utility scripts (data preprocessing, evaluation)
├── src/                        # Source code for model, utilities
├── .gitattributes              # For Git LFS tracking
├── .gitignore
└── README.md
```

## 🙏 Acknowledgements
*   Mention any datasets, pre-trained models, or codebases that were particularly helpful.
*   Thank you to my Data Chef teammates: Darren, Freddie, and Felix, for whom without this challenge would not have been possible.
