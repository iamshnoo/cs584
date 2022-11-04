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
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
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
    train_dataset.sparsity = train_dataset.X.nnz / (
        train_dataset.X.shape[0] * train_dataset.X.shape[1]
    )
    print("-" * 80)
    print(f"X shape: {train_dataset.X.shape}")
    print(f"Number of total samples: {len(train_dataset.y)}")
    print(f"Balanced Class distribution after SMOTE : {train_dataset.dist}")
    print(f"Sparsity : {train_dataset.sparsity}")
    print("-" * 80)

    """
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
    print("-" * 80)
    plt.plot(p, num_features)
    plt.xlabel("p")
    plt.ylabel("Number of features")
    plt.title("p vs Number of features")
    plt.savefig("../figs/p_vs_num_features.png")
    plt.show()
    """

    """
    # create a selectkbest object
    # select the k best features according to the chi-squared test
    # search over k values between 1000 and 10000
    # to find the best value for the number of features to be selected
    k = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000]
    f1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    clf = Classifier(
            MultinomialNB(alpha=0.7, fit_prior=True, class_prior=[0.2, 0.8]),
            "MultinomialNB",
    )
    for num_features in sorted(k):
        print(f"Finding best number of features to be selected: {num_features}")
        chi2_selector = SelectKBest(chi2, k=num_features)
        chi2_selector.fit(train_dataset.X, train_dataset.y)
        #train_dataset.X = chi2_selector.transform(train_dataset.X)
        X = chi2_selector.transform(train_dataset.X)
        print(f"X shape after SelectKBest feature selection: {X.shape}")
        print("-" * 80)
        # cross validate and find best k according to f1 score on the classifier
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        scores = cross_validate(
            clf.model,
            X,
            train_dataset.y,
            scoring=["f1", "precision", "recall"],
            cv=cv,
            n_jobs=-1,
        )

        f1_score = np.mean(scores["test_f1"])
        f1[k.index(num_features)] = f1_score

        # plot k vs f1 score
        print("Plotting k vs f1 score...")
        plt.plot(k, f1)
        plt.xlabel("k")
        plt.ylabel("f1 score")
        plt.title("k vs f1 score")
        plt.savefig("../figs/select_kbest_k_vs_f1_score.png")
        plt.show()

        print(f1)
        print(f"Mean f1 score: {f1_score}")
        print("-" * 80)
        # choose the best k
        best_k = k[f1.index(max(f1))]
        print(f"Best k: {best_k}")
        print("-" * 80)
    """

    """
    # plot the 100 most important features
    fig, ax = plt.subplots()
    # set figure size
    fig.set_size_inches(20, 10)
    # choosing 100 features (other features have very low importance < 0.002)
    ax = sns.barplot(x=forest_importances.nlargest(100), y=forest_importances.nlargest(100).index)
    #forest_importances.nlargest(250).plot.bar(ax=ax)
    ax.set_title("Top 100 features")
    ax.set_yticklabels(forest_importances.nlargest(100).index)
    ax.set_xlabel("Feature importance score")
    ax.set_ylabel("Features")
    fig.tight_layout()
    plt.savefig("../figs/feature_importances_random_forest.png")
    plt.show()

    # print the 100 most important features
    print("100 most important features:")
    print(forest_importances.nlargest(100))
    print("-" * 80)
    """

    print("Finding best variance threshold...")
    p = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    num_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(p)):
        bernoulli = VarianceThreshold(threshold=(p[i] * (1 - p[i])))
        bernoulli.fit(train_dataset.X)
        num_features[i] = len(bernoulli.get_support(indices=True))
        print(f"p: {p[i]}, Number of features to be selected:{num_features[i]}")
    print("-" * 80)

    p = 0.99  # eliminate features that are either zero or one in more than "p" percent of the samples
    sel = VarianceThreshold(threshold=(p * (1 - p)))
    train_dataset.X = sel.fit_transform(train_dataset.X)
    print(f"X shape after feature selection: {train_dataset.X.shape}")
    print("-" * 80)

    path1 = "../outputs/val_results_14k_features_ablation.csv"
    path2 = "../figs/val_f1_score_14k_features_ablation.png"
    path3 = "../figs/validation_results_14k_features_ablation.png"
    path4 = "../outputs/test_predictions_14k_features_ablation.csv"

    model = "decision tree"

    if model == "NB":
        path5 = "../outputs/grid_search_multinomial_nb_ablation.csv"
        path6 = "../figs/grid_search_multinomial_nb_ablation.png"

        # clf = Classifier(MultinomialNB(), "MultinomialNB")

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        p_grid_NB = {
            "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "fit_prior": [True, False],
            "class_prior": [
                None,
                [0.1, 0.9],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
                [0.5, 0.5],
                [0.6, 0.4],
                [0.7, 0.3],
                [0.8, 0.2],
                [0.9, 0.1],
            ],
        }

        grid = GridSearchCV(
            estimator=MultinomialNB(),
            param_grid=p_grid_NB,
            scoring="f1_macro",
            cv=cv,
            verbose=2,
            n_jobs=-1,
        )
        grid.fit(
            train_dataset.X,
            train_dataset.y,
        )

        # export grid results to dataframe
        df = pd.DataFrame(grid.cv_results_)
        # rank by mean test score
        df = df.sort_values(by=["rank_test_score"])
        df.to_csv(path5, index=False)

        # print best score and best params
        print(f"Best score: {grid.best_score_}")
        print(f"Best params: {grid.best_params_}")

        # plot grid search results
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.lineplot(
            x="param_alpha",
            y="mean_test_score",
            hue="param_fit_prior",
            data=df,
            ax=ax,
        )
        ax.set_title("Grid Search Results")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("F1 Score")
        ax.legend(title="Fit Prior")
        plt.savefig(path6)
        plt.show()

    elif model == "perceptron":
        path5 = "../outputs/grid_search_perceptron_ablation.csv"
        path6 = "../figs/grid_search_perceptron_ablation.png"

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        p_grid_perceptron = {
            "penalty": ["l1", "l2", "elasticnet"],
            "alpha": [0.01, 0.1, 1.0],
            "fit_intercept": [True, False],
            "max_iter": [3000, 5000],
            "tol": [1e-3, 1e-4, 1e-5],
            "shuffle": [True, False],
            "eta0": [0.1, 0.01],
            "class_weight": [
                {0: 0.1, 1: 0.9},
                {0: 0.2, 1: 0.8},
                {0: 0.3, 1: 0.7},
                {0: 0.4, 1: 0.6},
                {0: 0.5, 1: 0.5},
            ],
        }

        grid = GridSearchCV(
            estimator=Perceptron(),
            param_grid=p_grid_perceptron,
            scoring="f1_macro",
            cv=cv,
            verbose=2,
            n_jobs=-1,
        )

        grid.fit(
            train_dataset.X,
            train_dataset.y,
        )

        # export grid results to dataframe
        df = pd.DataFrame(grid.cv_results_)
        # rank by mean test score
        df = df.sort_values(by=["rank_test_score"])
        df.to_csv(path5, index=False)

        # print best score and best params
        print(f"Best score: {grid.best_score_}")
        print(f"Best params: {grid.best_params_}")

        # plot grid search results
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.lineplot(
            x="param_alpha",
            y="mean_test_score",
            hue="param_penalty",
            data=df,
            ax=ax,
        )
        ax.set_title("Grid Search Results")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("F1 Score")
        ax.legend(title="Penalty")
        plt.savefig(path6)
        plt.show()

    elif model == "decision tree":
        path5 = "../outputs/grid_search_decision_tree_ablation.csv"
        path6 = "../figs/grid_search_decision_tree_ablation.png"

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        p_grid_decision_tree = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [4, 5, 6, 7, 8, 9, 10],
            "min_samples_split": [4, 5, 6, 7, 8, 9, 10],
        }

        grid = GridSearchCV(
            estimator=DecisionTreeClassifier(),
            param_grid=p_grid_decision_tree,
            scoring="f1_macro",
            cv=cv,
            verbose=2,
            n_jobs=-1,
        )

        grid.fit(
            train_dataset.X,
            train_dataset.y,
        )

        # export grid results to dataframe
        df = pd.DataFrame(grid.cv_results_)
        # rank by mean test score
        df = df.sort_values(by=["rank_test_score"])
        df.to_csv(path5, index=False)

        # print best score and best params
        print(f"Best score: {grid.best_score_}")
        print(f"Best params: {grid.best_params_}")

        # plot grid search results
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.lineplot(
            x="param_max_depth",
            y="mean_test_score",
            hue="param_criterion",
            data=df,
            ax=ax,
        )
        ax.set_title("Grid Search Results")
        ax.set_xlabel("Max Depth")
        ax.set_ylabel("F1 Score")
        ax.legend(title="Criterion")
        plt.savefig(path6)
        plt.show()
