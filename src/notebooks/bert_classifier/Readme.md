# Fine-tuning BERT for Rumor Verification

This project fine-tunes a BERT model for rumor verification using both training and validation datasets. The model classifies rumors into three categories:

- **SUPPORTS**: The rumor is supported by evidence.
- **REFUTES**: The rumor is refuted by evidence.
- **NOT ENOUGH INFO**: There is not enough information to classify the rumor.

## Workflow Overview

### 1. Data Loading and Preprocessing
- The training and validation datasets are loaded from JSON files, containing **rumor**, **timeline**, and **evidence** data.
- The `Preprocessor` class is used to clean the text by removing URLs, special characters, stopwords, and emojis.
- Both the training and validation datasets are preprocessed to prepare them for feature extraction and model input.

### 2. Feature Extraction (Evidence Retrieval)
- **BERT Embeddings**: BERT is used as a feature extractor through the `FeatureExtractor` class. It generates embeddings for the rumors and the timeline entries.
- **Cosine Similarity**: The similarity between the rumor and timeline entries is calculated using cosine similarity.
- **Top-N Evidence Selection**: The top N timeline entries most similar to the rumor are selected as evidence for that rumor. This evidence is combined with the rumor text for model classification.

### 3. Rumor Classification
- The combined rumor and evidence text is tokenized using BERT’s tokenizer.
- A custom `RumorDataset` class is used to prepare the tokenized data for input into the BERT classification model.
- Labels are mapped to integer values:
  - **0**: REFUTES
  - **1**: SUPPORTS
  - **2**: NOT ENOUGH INFO

### 4. Model Evaluation
- A pre-trained BERT model is fine-tuned for rumor classification.
- The Hugging Face `Trainer` API is used for evaluation, providing support for batch processing and metric calculations.
- The model is evaluated using metrics such as **precision**, **recall**, **F1 score**, and **accuracy**.

### 5. Model Saving
- After training, the model and tokenizer are saved to disk for future use.

## Key Components
      V
### 1. Preprocessor
Cleans and preprocesses the text data by removing noise like URLs, emojis, special characters, and stopwords from the rumor, evidence, and timeline entries.

### 2. FeatureExtractor
Extracts BERT embeddings for the rumor and timeline entries. These embeddings are used to compute cosine similarity between the rumor and timeline entries to find the most relevant evidence.

### 3. RumorDataset
A custom PyTorch dataset that tokenizes the combined rumor and evidence text and prepares it for input into BERT for sequence classification.

### 4. Trainer
The Hugging Face `Trainer` API manages the evaluation of the BERT model, including batch processing and metric calculations.

### 5. Metrics Calculation
**Precision**, **recall**, **F1 score**, and **accuracy** are calculated using **scikit-learn** metrics.
