from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_validate

from data_loader import Dataset

# set sns style
sns.set_style("darkgrid")
sns.set_palette("pastel")


def print_shapes(train_dataset: Dataset, test_dataset: Dataset):
    print("-" * 80)
    print(f"Train data shape: {train_dataset.X.shape}")
    print(f"Number of total samples: {len(train_dataset.y)}")
    print(f"Balanced Class distribution after SMOTETomek : {train_dataset.dist}")
    print(f"Sparsity : {train_dataset.sparsity}")
    print("-" * 80)
    # print(f"Train data shape after feature selection: {train_dataset.X.shape}")
    print("-" * 80)
    print("Test set after feature selection ")
    print("-" * 80)
    print(f"X shape: {test_dataset.X.shape}")
    print(f"Number of total samples: {len(test_dataset.y)}")
    print("-" * 80)


def print_test_set_dist(test_dataset, y_pred):
    print(f"Predicted class distribution of test set: {test_dataset.dist}")
    print("-" * 80)
    print("Test set results")
    print("-" * 80)
    print(y_pred)
    print("-" * 80)


def cv_loop(stump, num_learners, n_splits, n_repeats, metric, train_dataset, clf, cv):
    train_mean_scores, test_mean_scores, train_std_scores, test_std_scores = (
        [],
        [],
        [],
        [],
    )
    best_k = 0
    print("-" * 80)
    for k in num_learners:
        clf.set_params(k=k, stump=stump)
        scores = cross_validate(
            clf,
            train_dataset.X,
            train_dataset.y,
            cv=cv,
            scoring=metric,
            return_train_score=True,
        )
        train_scores = scores["train_score"]
        test_scores = scores["test_score"]
        train_scores = train_scores.reshape(n_repeats, n_splits)
        test_scores = test_scores.reshape(n_repeats, n_splits)

        train_mean_scores.append(np.mean(scores["train_score"], axis=0))
        test_mean_scores.append(np.mean(scores["test_score"], axis=0))
        train_std_scores.append(np.std(scores["train_score"], axis=0))
        test_std_scores.append(np.std(scores["test_score"], axis=0))

        print("k = ", k)
        print(
            "Train f1_score = ",
            np.mean(scores["train_score"], axis=0),
            "+/-",
            np.std(scores["train_score"], axis=0),
        )
        print(
            "Test f1_score = ",
            np.mean(scores["test_score"], axis=0),
            "+/-",
            np.std(scores["test_score"], axis=0),
        )
        print("-" * 80)

        if np.mean(scores["test_score"], axis=0) > np.mean(test_mean_scores, axis=0):
            best_k = k

    # deal with a case for linear svm, where best k comes out to be 0
    if best_k == 0:
        best_k = num_learners[0]
    return (
        train_mean_scores,
        test_mean_scores,
        train_std_scores,
        test_std_scores,
        best_k,
    )


def cv_plot(
    stump,
    cv_f1_plot_path,
    num_learners,
    metric,
    train_mean_scores,
    test_mean_scores,
    train_std_scores,
    test_std_scores,
):
    plt.errorbar(num_learners, train_mean_scores, yerr=train_std_scores, label="train")
    plt.errorbar(num_learners, test_mean_scores, yerr=test_std_scores, label="test")
    plt.title(f"Cross Validation scores : ({metric} metric, {stump} adaboost decision stump) ")
    plt.xlabel("Number of learners")
    plt.ylabel(metric)
    plt.legend()
    plt.axis("tight")
    plt.savefig(cv_f1_plot_path)
    plt.show()


def test_predictions(stump, final_predictions_path, train_dataset, test_dataset, clf, best_k):
    clf.set_params(k=best_k, stump=stump)
    clf.fit(train_dataset.X, train_dataset.y)
    y_pred = clf.predict(test_dataset.X)
    test_dataset.dist = Counter(y_pred)
    pd.DataFrame(y_pred).to_csv(
        final_predictions_path,
        index=False,
        header=False,
    )
    return y_pred
