import re
import spacy
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Initialize VADER for sentiment analysis
sentiment_analyzer = SentimentIntensityAnalyzer()

class AdditionalFeatureExtractor:
    def __init__(self, 
                 use_ner=True, 
                 use_sentiment=True, 
                 use_emoji_embeddings=False, 
                 emoji_extractor=None,
                 use_url=True, 
                 use_hashtags=True, 
                 use_mentions=True,
                 use_keywords=True):  
        self.use_ner = use_ner
        self.use_sentiment = use_sentiment
        self.use_emoji_embeddings = use_emoji_embeddings
        self.emoji_extractor = emoji_extractor  
        self.use_url = use_url
        self.use_hashtags = use_hashtags
        self.use_mentions = use_mentions
        self.use_keywords = use_keywords  

    def extract_ner(self, text):
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_sentiment(self, text):      
        sentiment_scores = sentiment_analyzer.polarity_scores(text)
        return sentiment_scores

    def extract_emoji_embeddings(self, text):
        emoji_embeddings = {}
        emojis_in_text = emoji.emoji_list(text)
        for emoji_info in emojis_in_text:
            emoji_char = emoji_info['emoji']
            embedding = self.emoji_extractor.get_emoji_embedding(emoji_char)
            if embedding is not None:
                emoji_embeddings[emoji_char] = embedding
        return emoji_embeddings

    def extract_url_features(self, text):      
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        urls = url_pattern.findall(text)
        return {
            "urls": urls
        }

    def extract_hashtags(self, text):
        hashtags = re.findall(r'#\w+', text)
        return {
            "hashtags": hashtags
        }

    def extract_mentions(self, text):
        mentions = re.findall(r'@\w+', text)
        return {
            "mentions": mentions
        }

    def extract_keywords(self, text):

        # Extract keywords from URLs and hashtags in the given text.
        
        url_keywords = []
        hashtag_keywords = []

        # Extract URL keywords
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        urls = url_pattern.findall(text)
        
        for url in urls:
            # Remove protocol and split the URL to extract meaningful words
            processed_url = re.sub(r'https?://(www\.)?', '', url)
            processed_url = re.sub(r'\W+', ' ', processed_url)  # Extract meaningful words
            url_keywords.extend(processed_url.split())

        # Extract Hashtags and remove `#` to create keywords
        hashtags = re.findall(r'#\w+', text)
        hashtag_keywords = [hashtag.strip('#') for hashtag in hashtags]

        # Combine keywords from URLs and hashtags
        combined_keywords = url_keywords + hashtag_keywords

        return {
            "urls": urls,
            "hashtags": hashtags,
            "combined_keywords": combined_keywords
        }

    def extract_features(self, text):
        features = {}

        if self.use_ner:
            features['ner'] = self.extract_ner(text)
        
        if self.use_sentiment:
            features['sentiment'] = self.extract_sentiment(text)
        
        if self.use_emoji_embeddings and self.emoji_extractor:
            features['emoji_embeddings'] = self.extract_emoji_embeddings(text)
        
        if self.use_url:
            features['url_features'] = self.extract_url_features(text)
        
        if self.use_hashtags:
            features['hashtags'] = self.extract_hashtags(text)
        
        if self.use_mentions:
            features['mentions'] = self.extract_mentions(text)

        if self.use_keywords:
            keyword_data = self.extract_keywords(text)
            features['keywords'] = keyword_data['combined_keywords']

        return features