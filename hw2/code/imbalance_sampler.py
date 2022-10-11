from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree

from data_loader import load_dataset
from model import Classifier

if __name__ == "__main__":
    path = "../data/train.txt"
    mode = "train"
    train_dataset = load_dataset(path, mode)
    # X, y = train_dataset.X, train_dataset.y
    # model = DecisionTreeClassifier()
    # # define resampling
    # #over = RandomOverSampler(sampling_strategy=0.9)
    # # over = RandomOverSampler(sampling_strategy=0.1)
    # # under = RandomUnderSampler(sampling_strategy=0.5)
    # # define pipeline
    # # pipeline = Pipeline(steps=[('o', over), ('u', under), ('m', model)])
    # # resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
    # resample = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
    # pipeline = Pipeline(steps=[('r', resample),('m', model)])
    # # define evaluation procedure
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # # evaluate model
    # scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # # summarize performance
    # print('Mean ROC AUC: %.3f' % mean(scores))
    print(f"X shape: {train_dataset.X.shape}")
    print(f"Number of total samples: {len(train_dataset.y)}")
    print(f"Original class distribution: {train_dataset.dist}")

    # sme = SMOTEENN(random_state=42)
    X_res, y_res = SMOTE().fit_resample(train_dataset.X, train_dataset.y)
    # X_res, y_res = sme.fit_resample(train_dataset.X, train_dataset.y)
    print(f"X_res shape: {X_res.shape}")
    print(f"Number of total samples: {len(y_res)}")
    print(f"Resampled class distribution: {Counter(y_res)}")

    # update the dataset
    train_dataset.X = X_res
    train_dataset.y = y_res
    train_dataset.dist = Counter(y_res)

    # create a list of classifiers
    classifiers = [
        Classifier(DummyClassifier(strategy="most_frequent"), "Dummy"),
        # Classifier(GaussianNB(), "Gaussian Naive Bayes"),
        Classifier(MultinomialNB(), "Multinomial Naive Bayes"),
        Classifier(ComplementNB(), "Complement Naive Bayes"),
        Classifier(BernoulliNB(), "Bernoulli Naive Bayes"),
        Classifier(DecisionTreeClassifier(), "Decision Tree"),
        Classifier(Perceptron(), "Perceptron"),
    ]
    print("-" * 80)
    for clf in classifiers:
        clf.fit(train_dataset.X, train_dataset.y)
        # print confusion matrix
        print(f"{clf.model_name} confusion matrix:")
        print(
            pd.crosstab(
                train_dataset.y,
                clf.predict(train_dataset.X),
                rownames=["Actual"],
                colnames=["Predicted"],
            )
        )
        # print accuracy
        print(
            f"{clf.model_name} accuracy: {clf.score(train_dataset.X, train_dataset.y)}"
        )
        # print f1 score
        print(
            f"{clf.model_name} f1 score: {clf.f1_score(train_dataset.X, train_dataset.y)}"
        )
        # print precision score
        print(
            f"{clf.model_name} precision score: {clf.precision_score(train_dataset.X, train_dataset.y)}"
        )
        # print recall score
        print(
            f"{clf.model_name} recall score: {clf.recall_score(train_dataset.X, train_dataset.y)}"
        )
        print("-" * 80)

    # plot the decision tree classifier
    plot_tree(classifiers[4].model)
    plt.savefig("../figs/decision_tree.png", dpi=800)
    plt.show()
