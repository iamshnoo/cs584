from collections import Counter
from typing import Any
import timeit

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from cross_validation import cv_loop, cv_plots
from data_loader import Dataset, load_dataset
from model import Classifier
from pre_process import PreProcess, imbalance_class_sampling
from utils import print_cv_results, print_shapes, print_test_set_dist

# set sns style
sns.set_style("darkgrid")
sns.set_palette("pastel")


def final_model_fitting(
    classifiers: list, train_dataset: Dataset, best_model: Any
) -> tuple:
    # fit the best model on the whole training set
    best_model.fit(train_dataset.X, train_dataset.y)
    best_model_params = best_model.get_params()
    print("Best model params")
    print("-" * 80)
    for key, value in best_model_params.items():
        print(f"{key}: {value}")
    print("-" * 80)

    ensemble = VotingClassifier(
        estimators=[
            ("perceptron", classifiers[0].model),
            ("multinomial_nb", classifiers[1].model),
            ("decision_tree", classifiers[2].model),
        ],
        voting="hard",
    )
    ensemble.fit(train_dataset.X, train_dataset.y)
    return best_model, ensemble


def predictions(test_dataset, best_model, ensemble):
    # predict the test set by considering agreement between best and ensemble
    y_pred_1 = best_model.predict(test_dataset.X)
    y_pred_2 = ensemble.predict(test_dataset.X)

    y_pred = np.array(
        [
            1 if y_pred_1[i] == 1 and y_pred_2[i] == 1 else 0
            for i in range(len(y_pred_1))
        ]
    )
    test_dataset.dist = Counter(y_pred)
    return y_pred


if __name__ == "__main__":

    start = timeit.default_timer()
    with open("configs.yaml", "r") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    stop = timeit.default_timer()
    execution_time = stop - start
    print("-" * 80)
    print(f"Configs loaded in {str(round(execution_time,4))} seconds")
    print("-" * 80)

    classifiers = [
        Classifier(
            Perceptron(
                alpha=CONFIG["perceptron_alpha"],
                class_weight={
                    0: CONFIG["perceptron_class_weight_0"],
                    1: CONFIG["perceptron_class_weight_1"],
                },
                eta0=CONFIG["perceptron_eta_0"],
                fit_intercept=CONFIG["perceptron_fit_intercept"],
                max_iter=CONFIG["perceptron_max_iter"],
                penalty=CONFIG["perceptron_penalty"],
                shuffle=CONFIG["perceptron_shuffle"],
                tol=CONFIG["perceptron_tol"],
                random_state=CONFIG["random_state"],
            ),
            "Perceptron",
        ),
        Classifier(
            MultinomialNB(
                alpha=CONFIG["multinomial_nb_alpha"],
                fit_prior=CONFIG["multinomial_nb_fit_prior"],
                class_prior=[
                    CONFIG["multinomial_nb_class_prior_0"],
                    CONFIG["multinomial_nb_class_prior_1"],
                ],
            ),
            "MultinomialNB",
        ),
        Classifier(
            DecisionTreeClassifier(
                min_samples_split=CONFIG["decision_tree_min_samples_split"],
                splitter=CONFIG["decision_tree_splitter"],
                random_state=CONFIG["random_state"],
            ),
            "Decision Tree",
        ),
    ]

    cv = RepeatedStratifiedKFold(
        n_splits=CONFIG["cross_val_n_splits"],
        n_repeats=CONFIG["cross_val_n_repeats"],
        random_state=CONFIG["random_state"],
    )

    start = timeit.default_timer()
    train_dataset = load_dataset(CONFIG["train_path"], "train")
    test_dataset = load_dataset(CONFIG["test_path"], "test")
    stop = timeit.default_timer()
    execution_time = stop - start
    print("-" * 80)
    print(f"Datasets (train and test) loaded in {str(round(execution_time,4))} seconds")
    print("-" * 80)

    start = timeit.default_timer()
    imbalance_class_sampling(train_dataset, CONFIG["imbalance_method"])
    stop = timeit.default_timer()
    execution_time = stop - start
    print("-" * 80)
    print(
        f"Imbalance class sampling completed in {str(round(execution_time,4))} seconds"
    )
    print("-" * 80)

    start = timeit.default_timer()
    pre_process = PreProcess(
        p=CONFIG["p"],
        k=CONFIG["k"],
        num_trees=CONFIG["num_trees"],
        num_ftrs_final=CONFIG["num_ftrs_final"],
    )
    pre_process.fit(train_dataset)
    pre_process.transform(test_dataset)
    stop = timeit.default_timer()
    execution_time = stop - start
    print("-" * 80)
    print(
        f"Dimensionality reduction completed in {str(round(execution_time,4))} seconds"
    )
    print("-" * 80)

    print_shapes(train_dataset, test_dataset)

    start = timeit.default_timer()
    results, best_f1_score, best_model = cv_loop(cv, classifiers, train_dataset)
    stop = timeit.default_timer()
    execution_time = stop - start
    print("-" * 80)
    print(f"Cross validation loop completed in {str(round(execution_time,4))} seconds")
    print("-" * 80)

    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    results_df.to_csv(CONFIG["cv_results_path"], index=False)

    print_cv_results(results_df)

    cv_plots(CONFIG["cv_f1_plot_path"], CONFIG["cv_results_plot_path"], results)

    start = timeit.default_timer()
    # ensemble is just the majority voting for all the three classifiers
    best_model, ensemble = final_model_fitting(classifiers, train_dataset, best_model)

    y_pred = predictions(test_dataset, best_model, ensemble)
    stop = timeit.default_timer()
    execution_time = stop - start
    print("-" * 80)
    print(
        f"Test set prediction generation completed in {str(round(execution_time,4))} seconds"
    )
    print("-" * 80)

    pd.DataFrame(y_pred).to_csv(
        CONFIG["final_predictions_path"], index=False, header=False
    )
    print_test_set_dist(test_dataset, y_pred)
