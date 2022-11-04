from data_loader import Dataset
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def cv_loop(cv, classifiers: list, dataset: Dataset) -> tuple:
    results, best_f1_score, best_model = [], 0, None
    for classifier in classifiers:
        scores = cross_validate(
            classifier.model,
            dataset.X,
            dataset.y,
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
    return results, best_f1_score, best_model


def cv_plots(path2, path3, results):
    # line plot of f1 scores of all models using seaborn
    fig_1 = sns.lineplot(x="model", y="f1_score", data=pd.DataFrame(results))
    # annotate the plot
    for line in range(len(results)):
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
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
            )

    fig_2.savefig(path3)
    plt.show()
