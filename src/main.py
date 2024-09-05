from data_loading import DataLoader_Data
from language_detector import LanguageDetector
from preprocessor import Preprocessor
from feature_extractor_ import FeatureExtractor
from similarity_calculator import SimilarityCalculator
from rumor_classifier import RumorClassifier
from evaluation import Evaluation
from sentiment_analyzer import SentimentAnalyzer
from ner_extractor import NERExtractor
import json

if __name__ == "__main__":
    file_path = '../data/raw/English_train.json'
    preprocessed_file_path = '../data/processed/English_train_preprocessed.json'
    evidence_similarity_results_file_path = '../data/similarity/English_train_evidence_similarity_results.json'
    timeline_similarity_results_file_path = '../data/similarity/English_train_timeline_similarity_results.json'

    # Load and preprocess the dataset
    data_loader = DataLoader_Data(file_path)
    language_detector = LanguageDetector()
    preprocessor = Preprocessor(
        language=language_detector.detect_language(data_loader.data[0]['rumor']),
        remove_urls=True,
        remove_special_characters=True,
        remove_stopwords=True,
        remove_noise_words=True,
        remove_emojis=True,
        apply_stemming=False,
        apply_lemmatization=True
    )
    
    # Preprocess data
    preprocessed_data = []
    for item in data_loader.data:
        item['rumor'] = preprocessor.preprocess_text(item['rumor'])
        for i, timeline_entry in enumerate(item['timeline']):
            item['timeline'][i][2] = preprocessor.preprocess_text(timeline_entry[2])
        for j, evidence_entry in enumerate(item['evidence']):
            item['evidence'][j][2] = preprocessor.preprocess_text(evidence_entry[2])
        preprocessed_data.append(item)
    
    # Save preprocessed data
    with open(preprocessed_file_path, 'w') as f:
        json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)

    # Sentiment Analysis and NER
    sentiment_analyzer = SentimentAnalyzer()
    ner_extractor = NERExtractor()

    for item in preprocessed_data:
        item['sentiment'] = sentiment_analyzer.analyze_sentiment(item['rumor'])
        item['entities'] = ner_extractor.extract_entities(item['rumor'])

    # Feature extraction using BERT
    extractor = FeatureExtractor(method='bert')
    all_texts = [item['rumor'] for item in preprocessed_data] + \
                [timeline_entry[2] for item in preprocessed_data for timeline_entry in item['timeline']] + \
                [evidence_entry[2] for item in preprocessed_data for evidence_entry in item['evidence']]
    vectors = extractor.fit_transform(all_texts)

    index = 0
    for item in preprocessed_data:
        item['rumor_vector'] = vectors[index].tolist() if extractor.method == 'bert' else vectors[index].toarray()[0].tolist()
        index += 1
        for timeline_entry in item['timeline']:
            timeline_entry.append(vectors[index].tolist() if extractor.method == 'bert' else vectors[index].toarray()[0].tolist())
            index += 1
        for evidence_entry in item['evidence']:
            evidence_entry.append(vectors[index].tolist() if extractor.method == 'bert' else vectors[index].toarray()[0].tolist())
            index += 1

    # Calculate similarities
    similarity_calculator = SimilarityCalculator()
    evidence_similarities = similarity_calculator.calculate_evidence_similarity(preprocessed_data, extractor)
    timeline_similarities = similarity_calculator.calculate_timeline_similarity(preprocessed_data)

    # Save similarity results
    with open(evidence_similarity_results_file_path, 'w') as f:
        json.dump(evidence_similarities, f, ensure_ascii=False, indent=4)
    with open(timeline_similarity_results_file_path, 'w') as f:
        json.dump(timeline_similarities, f, ensure_ascii=False, indent=4)

    # Calculate averages for classification thresholds
    classifier = RumorClassifier(
        avg_total=similarity_calculator.calculate_average_similarity(evidence_similarities),
        avg_refutes=similarity_calculator.calculate_average_similarity([sim for sim in evidence_similarities if sim['label'] == "REFUTES"]),
        avg_supports=similarity_calculator.calculate_average_similarity([sim for sim in evidence_similarities if sim['label'] == "SUPPORTS"])
    )

    # Classify the rumors
    predictions = classifier.classify(timeline_similarities, preprocessed_data)

    # Evaluate with precision, recall, and F1
    ground_truth_labels = [item['label'] for item in preprocessed_data]
    evaluator = Evaluation(ground_truth_labels, predictions)
    metrics = evaluator.all_metrics()

    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
