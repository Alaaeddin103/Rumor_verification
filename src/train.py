import json
import torch
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

class RumorStanceTrainer:
    def __init__(self, model_name='xlm-roberta-base', num_labels=3, output_dir='/Users/alaaeddinalia/Desktop/thesis _Rumor_verifiction/Rumor_verification/src/traind_Models/fine-tuned-xlm-roberta'):
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        rumor_texts = [item['rumor'] for item in data]
        evidence_texts = [item['evidence'] for item in data]
        labels = [item['label'] for item in data]
        label_to_id = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
        numerical_labels = [label_to_id[label] for label in labels]
        combined_texts = [f"Rumor: {rumor} [SEP] Evidence: {evidence}" for rumor, evidence in zip(rumor_texts, evidence_texts)]
        return combined_texts, numerical_labels

    def tokenize_function(self, texts):
        return self.tokenizer(texts, padding="max_length", truncation=True, max_length=256)

    def create_dataset(self, texts, labels):
        encodings = self.tokenize_function(texts)
        labels_tensor = torch.tensor(labels)
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

            def __len__(self):
                return len(self.labels)
        return Dataset(encodings, labels_tensor)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    def train(self, data_path, num_train_epochs=10, per_device_train_batch_size=8):
        combined_texts, numerical_labels = self.load_data(data_path)
        train_texts, test_texts, train_labels, test_labels = train_test_split(combined_texts, numerical_labels, test_size=0.2)
        train_dataset = self.create_dataset(train_texts, train_labels)
        test_dataset = self.create_dataset(test_texts, test_labels)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            logging_dir='./logs',
            evaluation_strategy="epoch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        results = trainer.evaluate()
        print(f"Results: {results}")
        
        return results
