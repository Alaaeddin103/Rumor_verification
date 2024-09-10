from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class FeatureExtractor:
    def __init__(self, method='bert', bert_model_name='bert-base-uncased', sbert_model_name='paraphrase-multilingual-MiniLM-L12-v2', batch_size=16, max_features=1000):
        self.method = method
        self.batch_size = batch_size
        
        if method == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.model = BertModel.from_pretrained(bert_model_name)
        elif method == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
        elif method == 'sbert':
            self.sbert_model = SentenceTransformer(sbert_model_name)
        else:
            raise ValueError("Method not supported. Choose 'bert', 'tfidf', or 'sbert/multilingual-sbert'.")

    def fit_transform(self, texts):
        if self.method == 'bert':
            return self._bert_vectorize(texts)
        elif self.method == 'tfidf':
            return self.vectorizer.fit_transform(texts)
        elif self.method == 'sbert':
            return self._sbert_vectorize(texts)

    def transform(self, texts):
        if self.method == 'bert':
            return self._bert_vectorize(texts)
        elif self.method == 'tfidf':
            return self.vectorizer.transform(texts)
        elif self.method == 'sbert':
            return self._sbert_vectorize(texts)

    def _bert_vectorize(self, texts):
        all_vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_vectors = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            all_vectors.extend(batch_vectors)
        return np.array(all_vectors)
    
    def _sbert_vectorize(self, texts):
        return self.sbert_model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True)