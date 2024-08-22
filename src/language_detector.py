from langdetect import detect, DetectorFactory


DetectorFactory.seed = 0

class LanguageDetector:
    @staticmethod
    def detect_language(text):
        try:
            language = detect(text)
            return 'arabic' if language == 'ar' else 'english'
        except:
            return 'unknown'
