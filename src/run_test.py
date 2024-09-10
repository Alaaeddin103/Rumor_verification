import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from collections import Counter

class RumorStanceTest:
    def __init__(self, model_path):
        
        # Load the fine-tuned model and tokenizer using the model path
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        self.model.eval()

        # mapping from class IDs to stance labels
        self.label_mapping = {
            0: "SUPPORTS",
            1: "REFUTES",
            2: "NOT ENOUGH INFO"
        }

    def predict_stance(self, rumor_text, timeline_text):
       
        input_text = f"Rumor: {rumor_text} [SEP] Evidence: {timeline_text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        
        with torch.no_grad():
            outputs = self.model(**inputs) 
            logits = outputs.logits  
            predicted_class_id = logits.argmax().item()  
        
        predicted_label = self.label_mapping[predicted_class_id]
        
        return predicted_label, logits

    def majority_voting(self, predictions):
        # Apply majority voting to a list of predictions to determine the final stance.
        counter = Counter(predictions)
        most_common = counter.most_common(1)
        return most_common[0][0]

    def predict_rumor_stance(self, rumor_text, relevant_timelines, threshold_t2=0.5):    
        # Predict the final stance for a rumor based on multiple timelines.
        predictions = []
        probabilities = []

        for timeline in relevant_timelines:
            # Predict stance for each timeline
            predicted_label, logits = self.predict_stance(rumor_text, timeline['timeline_text'])
            predicted_class_id = logits.argmax().item()
            predictions.append(predicted_class_id)

            # Calculate softmax probabilities
            probs = F.softmax(logits, dim=1)
            predicted_class_prob = probs[0][predicted_class_id].item()
            probabilities.append(predicted_class_prob)

        # Apply majority voting to determine the final stance prediction
        final_prediction_id = self.majority_voting(predictions)
        final_label = self.label_mapping[final_prediction_id]

        # Filter relevant evidence based on class probability and threshold t2
        relevant_evidence = [
            (relevant_timelines[i], probabilities[i])
            for i in range(len(relevant_timelines))
            if predictions[i] == final_prediction_id and probabilities[i] >= threshold_t2
        ]

        # Sort relevant evidence by probability in descending order
        relevant_evidence_sorted = sorted(relevant_evidence, key=lambda x: x[1], reverse=True)

        return final_label, relevant_evidence_sorted
