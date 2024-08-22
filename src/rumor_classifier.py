from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class RumorClassifier:
    def __init__(self, avg_total, avg_refutes, avg_supports):
        self.avg_total = avg_total
        self.avg_refutes = avg_refutes
        self.avg_supports = avg_supports

    def classify(self, timeline_similarities, data):
        predictions = []

        for item in data:
            # Filter and sort similarities for the current rumor
            rumor_similarities = [sim for sim in timeline_similarities if sim['rumor_id'] == item['id']]
            top_5_similarities = sorted(rumor_similarities, key=lambda x: x['similarity'], reverse=True)[:5]

            # Filter similarities greater than avg_total
            filtered_similarities = [sim for sim in top_5_similarities if sim['similarity'] > self.avg_total]

            # Calculate the average of the filtered similarities
            avg_top_5_similarity = self.calculate_average_similarity(filtered_similarities)

            # Classify based on the average similarity
            if avg_top_5_similarity < self.avg_total:
                predictions.append("NOT ENOUGH INFO")
            elif abs(avg_top_5_similarity - self.avg_refutes) < abs(avg_top_5_similarity - self.avg_supports):
                predictions.append("REFUTES")
            else:
                predictions.append("SUPPORTS")

        return predictions  

    @staticmethod
    def calculate_average_similarity(similarities):
        if len(similarities) == 0:
            return 0.0
        total_similarity = sum([sim['similarity'] for sim in similarities])
        average_similarity = total_similarity / len(similarities)
        return average_similarity

    @staticmethod
    def evaluate(predictions, ground_truth):
        precision = precision_score(ground_truth, predictions, average='macro', zero_division=1)
        recall = recall_score(ground_truth, predictions, average='macro', zero_division=1)
        f1 = f1_score(ground_truth, predictions, average='macro', zero_division=1)
        strict_f1 = f1_score(ground_truth, predictions, average='macro', zero_division=1)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'strict_f1': strict_f1
        }
