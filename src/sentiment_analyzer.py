from textblob import TextBlob

class SentimentAnalyzer:
    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
