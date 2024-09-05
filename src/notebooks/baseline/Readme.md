# Rumor Verification System

## Overview

This notebook provides a pipeline for verifying rumors using evidence retrieved from authoritative Twitter timelines. The system combines various Natural Language Processing (NLP) techniques, including data preprocessing, sentiment analysis, Named Entity Recognition (NER), feature extraction (using BERT or TF-IDF), similarity calculation, rumor classification, and evaluation.

## System Workflow

1. **Data Loading and Preprocessing**
    - The dataset consists of rumors, timelines, and evidence.
    - A `LanguageDetector` detects the language (default is English).
    - The `Preprocessor` is used to clean the texts, including removing URLs, special characters, stopwords, and emojis, as well as applying lemmatization or stemming.
    - The dataset is preprocessed for rumors, timelines, and evidence, and saved to a JSON file for further processing.

2. **Sentiment Analysis and Named Entity Recognition (NER)**
    - `SentimentAnalyzer` is used to determine the sentiment of each rumor (positive, negative, or neutral).
    - `NERExtractor` is applied to identify and extract named entities (e.g., people, locations, organizations) from the rumor text.
    - Sentiment and entity information is stored alongside each rumor.

3. **Feature Extraction using TF-IDF or BERT**
    - `FeatureExtractor` is used to convert the rumor, timeline, and evidence texts into numerical feature vectors.
    - The feature extraction can be performed using TF-IDF (shallow, bag-of-words style) or BERT (deep contextualized embeddings).
    - These vectors are used for similarity calculations and rumor classification.

4. **Similarity Calculation**
    - The `SimilarityCalculator` calculates the cosine similarity between:
      - The rumor and evidence: Used to assess whether the evidence supports or refutes the rumor.
      - The rumor and timeline entries: Helps identify relevant information in the timeline.
    - The results are saved to JSON files for analysis and evaluation.

5. **Threshold Calculation for Classification**
    - Average similarity thresholds are calculated based on the evidence and timelines:
      - `avg_total`: Overall average similarity.
      - `avg_refutes`: Average similarity with refuting evidence.
      - `avg_supports`: Average similarity with supporting evidence.
    - These thresholds guide the classification of rumors.

6. **Rumor Classification**
    - The `RumorClassifier` uses the similarity thresholds to classify each rumor as either:
      - **SUPPORTS**: If the similarity with supporting evidence is high.
      - **REFUTES**: If the similarity with refuting evidence is high.
      - **NOT ENOUGH INFO**: If there is insufficient evidence to classify.
      
7. **Evaluation**
    - The system is evaluated using precision, recall, and F1-score metrics.
    - `Evaluation` compares the predicted labels with ground truth labels to provide insights into the system's performance.

## Key Files

- `data_loader.py`: Loads and parses the raw dataset (JSON format).
- `preprocessor.py`: Cleans and preprocesses the text (removal of noise, special characters, etc.).
- `feature_extractor_.py`: Handles the extraction of features using TF-IDF or BERT embeddings.
- `similarity_calculator.py`: Calculates cosine similarities between rumors, evidence, and timeline entries.
- `rumor_classifier.py`: Uses similarity scores and thresholds to classify rumors as **SUPPORTS**, **REFUTES**, or **NOT ENOUGH INFO**.
- `evaluation.py`: Provides evaluation metrics (precision, recall, F1-score) for the classification.

## Setup Instructions

1. **Install Dependencies**

   Run the following command to install the required Python libraries:

   ```bash
   pip install -r requirements.txt
