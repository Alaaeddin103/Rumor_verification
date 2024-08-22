from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)

    def fit_transform(self, texts):
        vectors = self.vectorizer.fit_transform(texts)
        return vectors

    def transform(self, texts):
        return self.vectorizer.transform(texts)
