import json

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data
    