from collections import Counter

from data_loader import load_dataset
from model import Classifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

import seaborn as sns

# set sns style
sns.set_style("darkgrid")
sns.set_palette("pastel")

def calc_cv_scores(train_dataset, classifiers, cv):
    roc_auc_results, names = [], []
    f1_results = []
    precision_results = []
    recall_results = []
    print("Evaluating models...")
    print("-"*80)
    # Ignore dummy when calculating scores
    for clf in classifiers:
        if clf.model_name == "Dummy":
            continue
        names.append(clf.model_name)
        if clf.model_name == "GaussianNB":
            X = np.asarray(train_dataset.X.todense())
        else:
            X = train_dataset.X
        for scoring in ["roc_auc", "f1_macro", "precision_macro", "recall_macro"]:
            scores = cross_val_score(clf.model, X, train_dataset.y, scoring=scoring, cv=cv, n_jobs=-1)
            if scoring == "roc_auc":
                roc_auc_results.append(scores)
                print(f"{clf.model_name}: ROC-AUC   Mean = {scores.mean():.3f}, Std = ({scores.std():.3f})")
            elif scoring == "f1_macro":
                f1_results.append(scores)
                print(f"{clf.model_name}: F1        Mean = {scores.mean():.3f}, Std = ({scores.std():.3f})")
            elif scoring == "precision_macro":
                precision_results.append(scores)
                print(f"{clf.model_name}: Precision Mean = {scores.mean():.3f}, Std = ({scores.std():.3f})")
            elif scoring == "recall_macro":
                recall_results.append(scores)
                print(f"{clf.model_name}: Recall    Mean = {scores.mean():.3f}, Std = ({scores.std():.3f})")
        print("-"*80)
    return roc_auc_results,names,f1_results,precision_results,recall_results

def box_plots_metric_comparison(roc_auc_results, names, f1_results, precision_results, recall_results):
    # create a subplot with 4 figures
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Model Comparison - Repeated Stratified K-Fold Cross Validation", fontsize=16)

    # plot roc auc results
    ax1.boxplot(roc_auc_results, labels=names)
    # label the scores
    for i in range(len(names)):
        ax1.text(i+1, roc_auc_results[i].mean(), f"{roc_auc_results[i].mean():.3f}", ha="center", va="bottom")

    ax1.set_title("ROC-AUC")
    # ax1.set_xlabel("Classifier")
    ax1.set_ylabel("ROC-AUC")

    # plot f1 macro results
    ax2.boxplot(f1_results, labels=names)
    # label the scores
    for i in range(len(names)):
        ax2.text(i+1, f1_results[i].mean(), f"{f1_results[i].mean():.3f}", ha="center", va="bottom")
    ax2.set_title("F1")
    # ax2.set_xlabel("Classifier")
    ax2.set_ylabel("F1")

    # plot precision macro results
    ax3.boxplot(precision_results, labels=names)
    # label the scores
    for i in range(len(names)):
        ax3.text(i+1, precision_results[i].mean(), f"{precision_results[i].mean():.3f}", ha="center", va="bottom")
    ax3.set_title("Precision")
    # ax3.set_xlabel("Classifier")
    ax3.set_ylabel("Precision")

    # plot recall macro results
    ax4.boxplot(recall_results, labels=names)
    # label the scores
    for i in range(len(names)):
        ax4.text(i+1, recall_results[i].mean(), f"{recall_results[i].mean():.3f}", ha="center", va="bottom")
    ax4.set_title("Recall")
    # ax4.set_xlabel("Classifier")
    ax4.set_ylabel("Recall")
    plt.savefig("../figs/model_comparison_box_plots.png")
    plt.show()

if __name__ == "__main__":
    path = "../data/train.txt"
    mode = "train"
    train_dataset = load_dataset(path, mode)
    X_res, y_res = SMOTE().fit_resample(train_dataset.X, train_dataset.y)
    train_dataset.X = X_res
    train_dataset.y = y_res
    train_dataset.dist = Counter(y_res)
    print("-"*80)
    print(f"X shape: {train_dataset.X.shape}")
    print(f"Number of total samples: {len(train_dataset.y)}")
    print(f"Balanced Class distribution after SMOTE : {train_dataset.dist}")
    print("-"*80)

    # create a list of classifiers
    classifiers = [
        Classifier(DummyClassifier(strategy="most_frequent"), "Dummy"),
        Classifier(Perceptron(), "Perceptron"),
        #Classifier(GaussianNB(), "GaussianNB"),
        Classifier(MultinomialNB(), "MultinomialNB"),
        Classifier(DecisionTreeClassifier(), "Decision Tree"),
    ]

    # stratified k-fold cross validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    # evaluate each model
    roc_auc_results, names, f1_results, precision_results, recall_results = calc_cv_scores(train_dataset, classifiers, cv)

    box_plots_metric_comparison(roc_auc_results, names, f1_results, precision_results, recall_results)
