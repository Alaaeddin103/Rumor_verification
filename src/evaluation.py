from sklearn.metrics import precision_score, recall_score, f1_score

class Evaluation:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def precision(self):
        return precision_score(self.y_true, self.y_pred, average='weighted')

    def recall(self):
        return recall_score(self.y_true, self.y_pred, average='weighted')

    def f1(self):
        return f1_score(self.y_true, self.y_pred, average='weighted')

    def all_metrics(self):
        return {
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1()
        }
