# Fine-tuning BERT for Rumor Verification

This notebook demonstrates how to fine-tune a BERT model for **rumor verification** using both training and validation datasets. The model is tasked with classifying rumors into one of three categories: **SUPPORTS**, **REFUTES**, or **NOT ENOUGH INFO**, based on the provided rumor and evidence.

## Workflow Overview

### 1. Data Loading and Preprocessing
- The data is loaded from **JSON** files, with each entry containing a **rumor**, **timeline**, and **evidence**.
- The `Preprocessor` is used to clean the text by:
  - Removing **URLs**, **special characters**, **stopwords**, and **emojis**.
- The rumors, evidence, and timelines are preprocessed for both the training and validation datasets.
- Labels are converted to integer values:
  - **0**: REFUTES
  - **1**: SUPPORTS
  - **2**: NOT ENOUGH INFO

### 2. Dataset Preparation
- The combined text (**rumor + evidence**) is tokenized using the **BERT tokenizer**.
- The tokenized data is stored in a custom `RumorDataset` class that prepares it for input into the BERT model.
- Each text is padded or truncated to a maximum length of **512 tokens**.

### 3. Model Initialization
- The BERT model is initialized for **sequence classification**, with 3 output labels:
  - SUPPORTS, REFUTES, and NOT ENOUGH INFO.
- The model is configured to run on **CPU** or **MPS** (if available).

### 4. Training and Evaluation
- The model is fine-tuned using the **Hugging Face Trainer API**.
- Training hyperparameters are defined using `TrainingArguments`, including:
  - **Learning rate**: 2e-5
  - **Number of epochs**: 10
  - **Batch size**: 8
  - **Evaluation strategy**: After every epoch
- Metrics such as **precision**, **recall**, **F1 score**, and **accuracy** are calculated using the **scikit-learn** library.

### 5. Model Saving
- After training, the model and tokenizer are saved to disk for future use.

## Key Files

- **`preprocessor.py`**: Handles text preprocessing tasks such as removing URLs, special characters, and stopwords.
- **`data_loading.py`**: Loads and parses the training and validation datasets.
- **`rumor_verification.ipynb`**: Main notebook that performs the training and evaluation of the BERT model.
