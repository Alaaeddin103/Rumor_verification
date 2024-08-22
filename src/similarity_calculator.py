from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimilarityCalculator:
    @staticmethod
    def calculate_cosine_similarity(vector1, vectors2):
        return cosine_similarity(vector1, vectors2)

    @staticmethod
    def calculate_average_similarity(similarities):
        if len(similarities) == 0:
            return 0.0
        total_similarity = sum([sim['similarity'] for sim in similarities])
        average_similarity = total_similarity / len(similarities)
        return average_similarity

    def calculate_evidence_similarity(self, data, vectorizer):
        evidence_similarities = []

        for item in data:
            rumor_vector = np.array(item['rumor_vector']).reshape(1, -1)
            evidence_vectors = [entry[3] for entry in item['evidence'] if len(entry) > 3 and entry[3]]
            
            if len(evidence_vectors) == 0:
                continue

            evidence_vectors = np.array(evidence_vectors)

            similarities = cosine_similarity(rumor_vector, evidence_vectors)

            for i, evidence_entry in enumerate(item['evidence']):
                if len(evidence_entry) > 3 and evidence_entry[3]:
                    similarity = similarities[0][i]
                    evidence_similarities.append({
                        'rumor_id': item['id'],
                        'evidence_id': evidence_entry[1],
                        'similarity': similarity,
                        'label': item['label']
                    })

        return evidence_similarities

    def calculate_timeline_similarity(self, data):
        timeline_similarities = []

        for item in data:
            rumor_vector = np.array(item['rumor_vector']).reshape(1, -1)
            timeline_vectors = np.array([entry[3] for entry in item['timeline'] if len(entry) > 3 and entry[3]])

            if len(timeline_vectors) == 0:
                continue

            similarities = cosine_similarity(rumor_vector, timeline_vectors)

            for i, timeline_entry in enumerate(item['timeline']):
                if len(timeline_entry) > 3 and timeline_entry[3]:
                    similarity = similarities[0][i]
                    timeline_similarities.append({
                        'rumor_id': item['id'],
                        'timeline_id': timeline_entry[1],
                        'similarity': similarity
                    })

        return timeline_similarities
