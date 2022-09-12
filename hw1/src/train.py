import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.utils import Bunch
from tqdm import tqdm

from knn import KNNClassifier
from pre_process import ecommerce_sentiment_analysis
from vectorize import vectorize

DEBUG = False


def k_fold_validation(dataset, kf, k, metric):
    all_y_pred = []
    all_y_test = []
    for train_index, test_index in kf.split(dataset.embeddings):
        X_train, X_test = (
            dataset.embeddings[train_index],
            dataset.embeddings[test_index],
        )
        y_train, y_test = (
            dataset.target[train_index],
            dataset.target[test_index],
        )
        if DEBUG:
            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        y_train = y_train.values
        y_test = y_test.values
        knn = KNNClassifier(k, metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        all_y_pred.append(y_pred)
        all_y_test.append(y_test)
    all_y_pred = np.concatenate(all_y_pred)
    all_y_test = np.concatenate(all_y_test)
    return np.sum(all_y_test == all_y_pred) / all_y_test.shape[0]


def grid_search_cv(parameter_list, dataset, kf):
    best = Bunch(k=1, accuracy=0, metric=None)
    logs = []
    for item in tqdm(
        parameter_list, desc="Grid search ...", total=len(parameter_list)
    ):
        k = item["n_neighbors"]
        metric = item["metric"]
        accuracy = k_fold_validation(dataset, kf, k, metric)

        # For two different metrics, if the best_accuracy is the same,
        # then the metric with the lower index in the metrics_list is chosen,
        # i.e. cosine is chosen over euclidean
        # because I am defining the relation below as strictly greater than.
        if accuracy > best.accuracy:
            best.accuracy = accuracy
            best.k = k
            best.metric = metric
            if DEBUG:
                print("-" * 80)
                print(f"best_k = {best.k}")
                print(f"best_accuracy = {best.accuracy}")
                print(f"best_metric = {best.metric}")
                print("-" * 80)
        logs.append([k, metric, accuracy])
    return best, logs


def plot_logs(logs, metrics_list, best):
    df = pd.DataFrame(logs, columns=["k", "metric", "accuracy"])
    # plot a 2D graph from logs of k and accuracy and color code the metric
    # also, highlight the best k
    fig, ax = plt.subplots()
    for metric in metrics_list:
        df_temp = df[df["metric"] == metric]
        ax.plot(df_temp["k"], df_temp["accuracy"], label=metric)
    ax.plot(
        best.k,
        best.accuracy,
        label="best_k",
        marker="o",
        markersize=5,
        color="red",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("accuracy")
    ax.set_title("k vs accuracy")
    ax.legend()
    plt.savefig("k_vs_accuracy.png", dpi=800)
    return fig


if __name__ == "__main__":

    # 1. Read raw data, process it and load it like an sklearn dataset
    path = "../data/train_file.csv"
    dev_percent = 0.001
    dataset = ecommerce_sentiment_analysis(path, dev_percent)

    # 2. Vectorize the dataset
    tf_idf = vectorize(dataset)

    # 3. Split the dataset into train and validation

    # 3.1 First, ensure that the splits are of equal size
    split_factor = 5
    num_splits = dataset.embeddings.shape[0]
    while num_splits >= 10:
        num_splits = num_splits // split_factor

    # 3.2 Then, create the kfold object
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    # 4. Grid search with k-fold-cross-validation for the KNN classifier
    # 4.1. Define the parameter grid
    # NOTE : constraints for k: k <= X_train.shape[0] and k % 2 == 1 (odd)
    k_list = (
        list(range(1, 13, 2))
        if dev_percent == 0.001
        else list(range(1, 500, 2))
    )
    metrics_list = ["cosine", "euclidean"]
    params = {"n_neighbors": k_list, "metric": metrics_list}
    parameter_list = list(ParameterGrid(params))
    assert len(parameter_list) == len(k_list) * len(metrics_list)

    # 4.2. Perform grid search
    best, logs = grid_search_cv(parameter_list, dataset, kf)

    # 5. Plot the results
    fig = plot_logs(logs, metrics_list, best)
    plt.show()
