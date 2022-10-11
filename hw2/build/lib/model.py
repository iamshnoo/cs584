from sklearn.metrics import f1_score, precision_score, recall_score

class Classifier:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        # calculate acuracy
        return self.model.score(X, y)

    def f1_score(self, X, y):
        # calculate f1 score
        return f1_score(y, self.predict(X), average="macro", zero_division=0)

    def precision_score(self, X, y):
        # calculate precision score
        return precision_score(y, self.predict(X), average="macro", zero_division=0)

    def recall_score(self, X, y):
        # calculate recall score
        return recall_score(y, self.predict(X), average="macro", zero_division=0)
