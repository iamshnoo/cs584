import argparse

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid

# import personal modules
from knn import KNNClassifier
from pre_process import ecommerce_sentiment_analysis, test_data
from train import grid_search_cv, plot_logs
from vectorize import reduce_dimensionality, vectorize


def main_tf_idf(config):
    dev_percent = config["dev_percent"]

    # 1. Read raw data, process it and load it like an sklearn dataset
    train_path = "../data/train_file.csv"
    train_dataset = ecommerce_sentiment_analysis(train_path, dev_percent)

    # 2. Vectorize the train_dataset
    tf_idf = vectorize(train_dataset, max_features=config["tf_idf_max_features"])
    print("TF-IDF Embeddings shape : ", train_dataset.embeddings.shape)

    # 2.1 Reduce dimensionality
    # lower the dimensionality, lower is my score on the leaderboard
    svd = reduce_dimensionality(train_dataset, config["svd_n_components"])
    print(
        "TF-IDF+Truncated SVD Embeddings shape : ",
        train_dataset.embeddings.shape,
    )

    # 3. Split the train_dataset into train and validation

    # 3.1 First, ensure that the splits are of equal size
    # also, I am choosing less than or equal to 10 splits
    split_factor = 5
    num_splits = train_dataset.embeddings.shape[0]
    while num_splits > 10:
        num_splits = num_splits // split_factor
    print("K-Fold num_splits : ", num_splits)

    # 3.2 Then, create the kfold object
    kf = (
        StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
        if config["kfold_stratified"]
        else KFold(n_splits=num_splits, shuffle=True, random_state=42)
    )

    # 4. Grid search with k-fold-cross-validation for the KNN classifier

    # 4.1. Define the parameter grid
    # NOTE 1 : constraints for k: k <= X_train.shape[0] and k % 2 == 1
    # NOTE 2 : to reduce search space, we increase the step size in the range
    # for k when dev_percent is 0
    # NOTE 3 : only metrics chosen are "cosine" and "euclidean"
    k_list = (
        list(range(1, 13, 2))
        if dev_percent == 0.001
        # else list(range(1, 500, 32))
        else list(
            range(
                config["k_list_min"],
                config["k_list_max"],
                config["k_list_step"],
            )
        )
    )
    metrics_list = ["cosine", "euclidean"]
    params = {"n_neighbors": k_list, "metric": metrics_list}
    parameter_list = list(ParameterGrid(params))
    assert len(parameter_list) == len(k_list) * len(metrics_list)

    # 4.2. Perform grid search
    best, logs = grid_search_cv(
        parameter_list,
        train_dataset,
        kf,
        weighted=config["weighted"],
        measure=config["measure"],
    )

    # 5. Plot the results
    _ = plot_logs(logs, metrics_list, best, name=config["plot_name"])

    # 6. Train the best model on the entire train_dataset
    #    and test it on the test_dataset

    # 6.1 Load test data
    test_path = "../data/test_file.csv"
    test_dataset = test_data(test_path, dev_percent)
    test_embeddings = tf_idf.transform(test_dataset.data)
    test_embeddings = svd.transform(test_embeddings)
    test_dataset.embeddings = test_embeddings

    X_train = train_dataset.embeddings
    y_train = train_dataset.target
    y_train = y_train.values
    X_test = test_dataset.embeddings

    print("X_train.shape", X_train.shape)
    print("y_train.shape", y_train.shape)
    print("X_test.shape", X_test.shape)
    # assert that X_train and X_test are not the same
    assert X_train is not X_test
    # assert that X_train and X_test have the same number of features
    assert X_train.shape[1] == X_test.shape[1]

    # 6.2 Train the best model on the entire train_dataset
    knn = KNNClassifier(
        n_neighbors=best.k, metric=best.metric, weighted=config["weighted"]
    )
    knn.fit(X_train, y_train)

    # 6.3 Test the best model on the test_dataset
    y_pred = knn.predict(X_test)
    # print("y_pred : ", y_pred)
    print("y_pred.shape : ", y_pred.shape)

    # 6.4 Save the predictions to a csv file
    df = pd.DataFrame(y_pred, columns=["target"])
    # if df["target"] is 1, then save it as "+1", else "-1" to match format file
    df["target"] = df["target"].apply(lambda x: "+1" if x == 1 else "-1")

    predictions_path = config["predictions_path"]
    df.to_csv(predictions_path, index=False, header=False)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    # dev percent
    parser.add_argument(
        "--dev_percent",
        type=float,
        default=0,
        help="Percent of data to load, 0 loads all data",
    )
    # tf_idf_max_features
    parser.add_argument(
        "--tf_idf_max_features",
        type=int,
        default=1000,
        help="Max features output for tf_idf, recommended 1000",
    )
    # svd_n_components
    parser.add_argument(
        "--svd_n_components",
        type=int,
        default=750,
        help="Number of components output for svd, recommended 750",
    )
    # kfold stratified
    parser.add_argument(
        "--kfold_stratified",
        type=bool,
        default=True,
        help="Whether to use stratified kfold",
    )
    # k_list_min
    parser.add_argument(
        "--k_list_min",
        type=int,
        default=1,
        help="Min value for k in list of values of k to iterate over",
    )
    # k_list_max
    parser.add_argument(
        "--k_list_max",
        type=int,
        default=1500,
        help="Max value for k in list of values of k to iterate over",
    )
    # k_list_step
    parser.add_argument(
        "--k_list_step",
        type=int,
        default=32,
        help="Step size for k in list of values of k to iterate over",
    )
    # weighted
    parser.add_argument(
        "--weighted",
        type=bool,
        default=True,
        help="Whether to use weighted kNN, recommended True",
    )
    # measure
    parser.add_argument(
        "--measure",
        type=str,
        default="f1",
        help="Measure to use for cross validation during grid search",
    )
    # plot_name
    parser.add_argument(
        "--plot_name",
        type=str,
        default="k_vs_accuracy_svd_750_tfidf_1000_k_1_1500_weighted_knn_f1_stratified_kfold.png",
        help="Name of plot to save",
    )
    # predictions_path
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="../predictions/predictions_svd_750_tfidf_1000_k_1_1500_weighted_knn_f1_stratified_kfold.csv",
        help="Path to save predictions",
    )

    args = parser.parse_args()

    # create a config dictionary to pass to the main function
    config = {
        "dev_percent": args.dev_percent,
        "tf_idf_max_features": args.tf_idf_max_features,
        "svd_n_components": args.svd_n_components,
        "kfold_stratified": args.kfold_stratified,
        "k_list_min": args.k_list_min,
        "k_list_max": args.k_list_max,
        "k_list_step": args.k_list_step,
        "weighted": args.weighted,
        "measure": args.measure,
        "plot_name": args.plot_name,
        "predictions_path": args.predictions_path,
    }

    main_tf_idf(config)
