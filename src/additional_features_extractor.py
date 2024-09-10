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
                 use_mentions=True):
        self.use_ner = use_ner
        self.use_sentiment = use_sentiment
        self.use_emoji_embeddings = use_emoji_embeddings
        self.emoji_extractor = emoji_extractor  # Pass an instance of EmojiEmbeddingExtractor
        self.use_url = use_url
        self.use_hashtags = use_hashtags
        self.use_mentions = use_mentions

    def extract_ner(self, text):
        # Extract Named Entities from the text using spaCy.
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_sentiment(self, text):      
        # Perform sentiment analysis using VADER.      
        sentiment_scores = sentiment_analyzer.polarity_scores(text)
        return sentiment_scores

    def extract_emoji_embeddings(self, text):
       
       # Extract emojis and their embeddings from the text.
        
        emoji_embeddings = {}
        emojis_in_text = emoji.emoji_list(text)  # Extract emojis from text

        for emoji_info in emojis_in_text:
            emoji_char = emoji_info['emoji']  # Get the emoji character
            embedding = self.emoji_extractor.get_emoji_embedding(emoji_char)
            if embedding is not None:
                emoji_embeddings[emoji_char] = embedding

        return emoji_embeddings

    def extract_url_features(self, text):      
        # Extract features from URLs in the text.       
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        urls = url_pattern.findall(text)
        return {
            "urls": urls
        }

    def extract_hashtags(self, text):
        # Extract hashtags from the text.
        hashtags = re.findall(r'#\w+', text)
        return {
            "hashtags": hashtags
        }

    def extract_mentions(self, text):
        # Extract @mentions from the text.
        mentions = re.findall(r'@\w+', text)
        return {
            "mentions": mentions
        }

    def extract_features(self, text):
        
        # Extract all selected features
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

        return features
