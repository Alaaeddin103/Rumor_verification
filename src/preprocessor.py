import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class Preprocessor:
    def __init__(self, language, remove_urls=False, remove_special_characters=False,
                 remove_stopwords=False, remove_noise_words=False, remove_emojis=False,
                 apply_stemming=False, apply_lemmatization=False, clean_special_characters=False):
        self.language = language
        self.remove_urls_flag = remove_urls
        self.remove_special_characters_flag = remove_special_characters
        self.remove_stopwords_flag = remove_stopwords
        self.remove_noise_words_flag = remove_noise_words
        self.remove_emojis_flag = remove_emojis
        self.apply_stemming_flag = apply_stemming
        self.apply_lemmatization_flag = apply_lemmatization
        self.clean_special_characters_flag = clean_special_characters
        
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
    
    def clean_special_characters(self, text):
        # Remove all characters except letters, numbers, spaces, @mentions, and hashtags
        return re.sub(r'[^a-zA-Z0-9\s@#]', '', text)

    def remove_special_characters(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def remove_noise_words(self, text):
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.noise_words]
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
        if self.remove_urls_flag:
            text = self.remove_urls(text)
            
        if self.remove_emojis_flag:
            text = self.remove_emojis(text)
            
        if self.remove_special_characters_flag:
            text = self.remove_special_characters(text)
            
        if self.clean_special_characters_flag:
            text = self.clean_special_characters(text)
                          
        if self.remove_stopwords_flag:
            text = self.remove_stopwords(text)
            
        if self.remove_noise_words_flag:
            text = self.remove_noise_words(text)

        if self.apply_stemming_flag:
            text = ' '.join([self.stemmer.stem(word) for word in word_tokenize(text)])
        
        if self.apply_lemmatization_flag:
            text = ' '.join([self.lemmatizer.lemmatize(word) for word in word_tokenize(text)])
        
        return text
