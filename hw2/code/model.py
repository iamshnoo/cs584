from data_loader import load_dataset

# import naive bayes classifier, decision tree classifier and neural network
# classifier from sklearn
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.dummy import DummyClassifier

# import f1 score from sklearn
from sklearn.metrics import f1_score, precision_score, recall_score

# import numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# import matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint

# set sns style
sns.set_style("darkgrid")
sns.set_palette("pastel")


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


if __name__ == "__main__":
    path = "../data/train.txt"
    mode = "train"
    train_dataset = load_dataset(path, mode)
    print(f"X shape: {train_dataset.X.shape}")
    print(f"Class distribution: {train_dataset.dist}")

    # create a list of classifiers
    classifiers = [
        # Classifier(DummyClassifier(strategy="most_frequent"), "Dummy"),
        Classifier(GaussianNB(), "GaussianNB"),
        # Classifier(MultinomialNB(), "Multinomial Naive Bayes"),
        # Classifier(ComplementNB(), "Complement Naive Bayes"),
        # Classifier(BernoulliNB(), "Bernoulli Naive Bayes"),
        # Classifier(DecisionTreeClassifier(), "Decision Tree"),
        # Classifier(Perceptron(), "Perceptron"),
    ]
    print("-" * 80)
    for clf in classifiers:
        if clf.model_name == "GaussianNB":
            X = np.asarray(train_dataset.X.todense())
        else:
            X = train_dataset.X
        clf.fit(X, train_dataset.y)

        # print confusion matrix
        print(f"{clf.model_name} confusion matrix:")
        pprint(
            pd.crosstab(
                train_dataset.y,
                clf.predict(X),
                rownames=["Actual"],
                colnames=["Predicted"],
            )
        )
        # print accuracy
        print(f"{clf.model_name} accuracy: {clf.score(X, train_dataset.y)}")
        # print f1 score
        print(f"{clf.model_name} f1 score: {clf.f1_score(X, train_dataset.y)}")
        # print precision score
        print(
            f"{clf.model_name} precision score: {clf.precision_score(X, train_dataset.y)}"
        )
        # print recall score
        print(f"{clf.model_name} recall score: {clf.recall_score(X, train_dataset.y)}")
        print("-" * 80)
