import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class Preprocessor:
    def __init__(self, language):
        self.language = language
        self.stop_words = self.load_stopwords(language)
        self.noise_words = self.load_noise_words(language)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def load_stopwords(self, language):
        return set(stopwords.words(language))

    def load_noise_words(self, language):
        if language == 'english':
            return {"rt", "via", "…"}
        elif language == 'arabic':
            return {"RT", "بواسطة", "…"}
        return set()

    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_special_characters(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords_and_noise(self, text):
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words and word.lower() not in self.noise_words]
        return ' '.join(filtered_words)

    def remove_emojis(self, text):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002700-\U000027BF"  # Dingbats
            u"\U000024C2-\U0001F251"  # Enclosed characters
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def preprocess_text(self, text):
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.remove_special_characters(text)
        text = self.remove_stopwords_and_noise(text)
        
        # Apply stemming
        text = ' '.join([self.stemmer.stem(word) for word in word_tokenize(text)])
        
        # Apply lemmatization
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in word_tokenize(text)])
        
        return text 