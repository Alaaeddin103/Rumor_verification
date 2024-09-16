import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MeanShift

class RelevantTimelineRetriever:
    def __init__(self, thresholding_technique='iqr', bandwidth=None, use_semi_quartile=False):
        self.thresholding_technique = thresholding_technique
        self.bandwidth = bandwidth
        self.use_semi_quartile = use_semi_quartile 

    def calculate_similarities(self, preprocessed_data):
        # Calculate cosine similarities between the rumor and its timeline entries.
        similarities = []
        
        for item in preprocessed_data:
            rumor_vector = np.array(item['rumor_vector'])

            # Calculate similarities with timeline entries
            for timeline_entry in item['timeline']:
                timeline_vector = np.array(timeline_entry[-1])
                similarity = cosine_similarity(rumor_vector.reshape(1, -1), timeline_vector.reshape(1, -1))[0][0]
                
                similarities.append({
                    'rumor_id': item['id'],
                    'rumor_text': item['rumor'],  
                    'timeline_id': timeline_entry[1],  
                    'timeline_text': timeline_entry[2], 
                    'similarity': similarity
                })
        
        return similarities

    def apply_iqr_threshold(self, similarities):
        similarities_list = [sim['similarity'] for sim in similarities]
        q1 = np.percentile(similarities_list, 25) 
        q3 = np.percentile(similarities_list, 75) 
        iqr = q3 - q1  

        # Use semi-quartile range if the flag is set
        if self.use_semi_quartile:
            lower_bound = q1 - 0.75 * iqr
            upper_bound = q3 + 0.75 * iqr
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

        # Filter similarities that fall within the IQR (or semi-quartile) bounds
        filtered_similarities = [sim for sim in similarities if lower_bound <= sim['similarity'] <= upper_bound]
        return filtered_similarities

    def apply_clustering(self, similarities):
        similarity_scores = np.array([sim['similarity'] for sim in similarities]).reshape(-1, 1)
        # Apply mean-shift clustering with the specified bandwidth
        clustering = MeanShift(bandwidth=self.bandwidth).fit(similarity_scores)
        cluster_labels = clustering.labels_

       
        largest_cluster = max(set(cluster_labels), key=list(cluster_labels).count)
        relevant_similarities = [sim for i, sim in enumerate(similarities) if cluster_labels[i] == largest_cluster]
        
        return relevant_similarities

    def apply_game_theory_threshold(self, similarities):
        similarities_list = [sim['similarity'] for sim in similarities]
        top_50_percent = np.percentile(similarities_list, 50)
        return [sim for sim in similarities if sim['similarity'] >= top_50_percent]


    def retrieve_relevant_timelines(self, timeline_similarities, data):
        relevant_timeline_entries = []

        for item in data:
            # Filter similarities for the current rumor
            rumor_similarities = [sim for sim in timeline_similarities if sim['rumor_id'] == item['id']]
            
            # Apply the chosen thresholding technique
            if self.thresholding_technique == 'iqr':
                filtered_similarities = self.apply_iqr_threshold(rumor_similarities)
            elif self.thresholding_technique == 'mean-shift':
                filtered_similarities = self.apply_clustering(rumor_similarities)
            elif self.thresholding_technique == 'game-theory':
                filtered_similarities = self.apply_game_theory_threshold(rumor_similarities)
            else:
                raise ValueError(f"Unknown thresholding technique: {self.thresholding_technique}")
            
            # Collect relevant timeline
            relevant_entries = [
                {
                    'timeline_id': sim['timeline_id'],
                    'timeline_text': sim['timeline_text']
                }
                for sim in filtered_similarities
            ]
            
            relevant_timeline_entries.append({
                'rumor_id': item['id'],
                'rumor_text': item['rumor'],
                'relevant_timelines': relevant_entries
            })
        
        return relevant_timeline_entries
