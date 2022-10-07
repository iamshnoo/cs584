from collections import Counter

import joblib

from data_loader import load_dataset
from model import Classifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold

import seaborn as sns

# set sns style
sns.set_style("darkgrid")
sns.set_palette("pastel")

if __name__ == "__main__":
    path = "../data/train.txt"
    mode = "train"
    train_dataset = load_dataset(path, mode)
    X_res, y_res = SMOTE().fit_resample(train_dataset.X, train_dataset.y)
    train_dataset.X = X_res
    train_dataset.y = y_res
    train_dataset.dist = Counter(y_res)
    print(f"Sparsity : {train_dataset.sparsity}")
    # update soarsity
    train_dataset.sparsity = train_dataset.X.nnz / (train_dataset.X.shape[0] * train_dataset.X.shape[1])
    print("-"*80)
    print(f"X shape: {train_dataset.X.shape}")
    print(f"Number of total samples: {len(train_dataset.y)}")
    print(f"Balanced Class distribution after SMOTE : {train_dataset.dist}")
    print(f"Sparsity : {train_dataset.sparsity}")
    print("-"*80)

    # FEATURE SELECTION

    # eliminate features that are either zero or one in more than "p" percent of
    # the samples
    # Boolean features are Bernoulli random variables
    # variance is given by p(1-p)
    # search over p values between 0.9 and 0.99
    # to find the best value for the variance threshold

    # plot p vs number of features
    print("Finding best variance threshold...")
    p = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    num_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(p)):
        bernoulli = VarianceThreshold(threshold=(p[i] * (1 - p[i])))
        bernoulli.fit(train_dataset.X)
        num_features[i] = len(bernoulli.get_support(indices=True))
        print(f"p: {p[i]}, Number of features to be selected:{num_features[i]}")
    print("-"*80)
    plt.plot(p, num_features)
    plt.xlabel("p")
    plt.ylabel("Number of features")
    plt.title("p vs Number of features")
    plt.savefig("../figs/p_vs_num_features.png")
    plt.show()

    p = 0.99 # eliminate features that are either zero or one in more than "p" percent of the samples
    sel = VarianceThreshold(threshold=(p * (1 - p)))
    train_dataset.X = sel.fit_transform(train_dataset.X)
    print(f"X shape after feature selection: {train_dataset.X.shape}")
    print("-"*80)

    path1 =  "../outputs/val_results_14k_features.csv"
    path2 = "../figs/val_f1_score_14k_features.png"
    path3 = "../figs/validation_results_14k_features.png"
    path4 = "../outputs/test_predictions_14k_features.csv"

    # create a list of classifiers
    classifiers = [
        Classifier(Perceptron(), "Perceptron"),
        Classifier(MultinomialNB(), "MultinomialNB"),
        Classifier(DecisionTreeClassifier(), "Decision Tree"),
    ]

    # stratified cross fold validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    # evaluate each model in turn based on accuracy, f1 score, precision and
    # recall
    # store the best in terms of f1 score
    # store all other results in a dataframe
    results = []
    best_f1_score = 0
    best_model = None
    for classifier in classifiers:
        scores = cross_validate(
            classifier.model,
            train_dataset.X,
            train_dataset.y,
            scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
            cv=cv,
            n_jobs=-1,
        )
        results.append(
            {
                "model": classifier.model_name,
                "accuracy": np.mean(scores["test_accuracy"]),
                "f1_score": np.mean(scores["test_f1_macro"]),
                "precision": np.mean(scores["test_precision_macro"]),
                "recall": np.mean(scores["test_recall_macro"]),
            }
        )
        if np.mean(scores["test_f1_macro"]) > best_f1_score:
            best_f1_score = np.mean(scores["test_f1_macro"])
            best_model = classifier.model

    # print the results
    print("Results")
    print("-"*80)
    print(pd.DataFrame(results).sort_values(by="f1_score", ascending=False))
    print("-"*80)

    # save the csv file
    pd.DataFrame(results).sort_values(by="f1_score", ascending=False).to_csv(
        path1, index=False
    )

    # line plot of f1 scores of all models using seaborn
    fig_1 = sns.lineplot(x="model", y="f1_score", data=pd.DataFrame(results))
    # annotate the plot
    for line in range(0, len(results)):
        fig_1.text(
            line,
            results[line]["f1_score"],
            round(results[line]["f1_score"], 4),
            horizontalalignment="center",
            size="medium",
            color="black",
            weight="semibold",
        )

    fig_1.set_title("F1 Score of all models")
    fig_1.set_xlabel("Model")
    fig_1.set_ylabel("F1 Score")
    fig_1.figure.savefig(path2)
    plt.show()

    # plot all the results
    fig_2, ax = plt.subplots(1, 4, figsize=(20, 5))
    for i, metric in enumerate(["accuracy", "f1_score", "precision", "recall"]):
        sns.barplot(x="model", y=metric, data=pd.DataFrame(results), ax=ax[i])
        ax[i].set_title(metric)
        # label the bars
        for p in ax[i].patches:
            ax[i].annotate(
                format(p.get_height(), ".4f"),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
            )

    fig_2.savefig(path3)
    plt.show()


    # fit the best model on the whole training set
    best_model.fit(train_dataset.X, train_dataset.y)
    # get the params of the best model
    best_model_params = best_model.get_params()
    # save the best model
    joblib.dump(best_model, "../outputs/best_model.pkl")

    # print the params
    print("Best model params")
    print("-"*80)
    for key, value in best_model_params.items():
        print(f"{key}: {value}")
    print("-"*80)

    # load the test set
    path = "../data/test.txt"
    mode = "test"
    test_dataset = load_dataset(path, mode)
    print("-"*80)
    print(f"X shape: {test_dataset.X.shape}")
    print(f"Number of total samples: {len(test_dataset.y)}")
    print("-"*80)

    p = 0.95 # eliminate features that are either zero or one in more than "p" percent of the samples
    test_dataset.X = sel.transform(test_dataset.X)

    # predict the test set
    y_pred = best_model.predict(test_dataset.X)
    # update the class distribution of the test set
    test_dataset.dist = Counter(y_pred)
    # print the class distribution of the test set
    print(f"Predicted class distribution of test set: {test_dataset.dist}")
    print("-"*80)
    print("Test set results")
    print("-"*80)
    print(y_pred)
    print("-"*80)

    # save the predictions to a csv file
    pd.DataFrame(y_pred).to_csv(path4, index=False, header=False)
