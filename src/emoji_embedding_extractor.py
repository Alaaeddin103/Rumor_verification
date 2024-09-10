import numpy as np

class EmojiEmbeddingExtractor:
    def __init__(self, emoji2vec_path):
        # Initialize the EmojiEmbeddingExtractor by loading emoji embeddings from a file.
        self.emoji_embeddings = self.load_emoji2vec(emoji2vec_path)
    
    def load_emoji2vec(self, path):
        # Load Emoji2Vec embeddings from a file.
        emoji_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                emoji_char = values[0] 
                vector = np.asarray(values[1:], dtype='float32')  
                emoji_dict[emoji_char] = vector
        return emoji_dict

    def get_emoji_embedding(self, emoji_char):
        # Get the embedding vector for a given emoji character.
        return self.emoji_embeddings.get(emoji_char, None)
