# Optical Character Recognition (OCR) Challenge - Data Chefs @ TIL-25 Hackathon

This repository (`lolkabash/til-25-data-chefs-OCR`) contains the code, models, notebooks, and scripts developed by **Team Data Chefs** for the Optical Character Recognition (OCR) challenge of the DSTA BrainHack TIL-AI 2025. Our goal was to accurately identify and extract text from various documents.

The development work involved a mix of Python scripting, Jupyter Notebooks for experimentation, and Shell scripts for automation.

## 📝 Challenge Description

The OCR challenge required us to develop a system capable of identifying and extracting textual content from a variety of document images.

## 🔗 Repository Links

*   **This Repository:** [lolkabash/til-25-data-chefs-OCR](https://github.com/lolkabash/til-25-data-chefs-OCR)
*   **Main Team Repository:** For an overview of our entire TIL-25 Hackathon project and other challenges, please visit [lolkabash/til-25-data-chefs](https://github.com/lolkabash/til-25-data-chefs).

## 💻 Key Technologies We Used

*   **PaddleOCR (v2.10.0):** Our core OCR engine for text detection, recognition, and layout analysis.
*   **Python:** The primary language for scripting, model training, and inference logic.
*   **Jupyter Notebooks:** Utilized extensively for experimentation, data exploration, and model development iterations.
*   **Shell Scripts:** Employed for automating tasks such as dataset management, training pipelines, and other repetitive processes.
*   **OpenCV (`opencv-python-headless`):** Leveraged for various image processing tasks.
*   **NumPy:** For efficient numerical computations, particularly with image data.
*   **Docker:** Used for creating consistent environments for dataset construction and model deployment (though specific Dockerfiles for this repo might be found in the main competition structure).
*   **FastAPI & Uvicorn:** Integrated for serving our OCR model as an API endpoint within the competition framework.

## ✨ Our Solution & Key Achievements

Our approach to the OCR challenge centered on leveraging the PaddleOCR framework, implementing a multi-stage pipeline for robust text extraction:

1.  **Layout Analysis:** We used the `picodet_lcnet_x1_0_fgd_layout_infer` model to understand the overall structure and organization of the documents.
2.  **Text Detection:** The `en_PP-OCRv3_det_infer` model was employed to accurately locate text regions within the images.
3.  **Text Direction Classification:** We utilized `ch_ppocr_mobile_v2.0_cls_infer` to determine the orientation of the detected text.
4.  **Text Recognition:** Initially, we used the `en_PP-OCRv4_rec_infer` model for converting image text into character strings.

### Finetuning for Enhanced Performance

To significantly boost the accuracy of the text recognition stage, we undertook a finetuning process for the `en_PP-OCRv4_rec_train` recognition model. This was a critical step in adapting the model to the specific characteristics of the competition dataset. Key aspects of our finetuning procedure included:

*   **Custom Character Set:** We curated a `custom_char_dict.txt` file, tailoring the vocabulary to the characters observed in the challenge data.
*   **Dataset:**
    *   Training Data: `rec_gt_train.txt`
    *   Evaluation Data: `rec_gt_eval.txt`
    *   Max Text Length: Configured to 128 characters.
*   **Training Parameters:**
    *   Epochs: 100
    *   Optimizer: Adam
    *   Learning Rate Scheduler: Cosine Annealing
    *   Initial Learning Rate: 0.0001
    *   Warmup: 2 epochs
    *   Batch Size: 64 per GPU
*   **Model Architecture:**
    *   Recognition Algorithm: SVTR (Vision Transformer)
    *   Backbone: MobileNetV1Enhance
    *   Head: CTCHead with CTCLoss
*   **Preprocessing:**
    *   Images were resized to `[3, 48, 320]` (Channels, Height, Width).
    *   Standard normalization techniques were applied.
*   **Initialization:** The finetuning process commenced from the `best_accuracy` checkpoint of the pre-trained `en_PP-OCRv4_rec_train` model.

This meticulous finetuning process was instrumental in improving our model's performance on the competition's OCR tasks, leading to more accurate text extraction. The scripts and notebooks related to this finetuning process can be found within this repository.
