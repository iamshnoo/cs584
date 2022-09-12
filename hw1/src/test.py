import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid

from knn import KNNClassifier
from pre_process import ecommerce_sentiment_analysis, test_data
from train import grid_search_cv, plot_logs
from vectorize import vectorize

if __name__ == "__main__":

    dev_percent = 0

    # 1. Read raw data, process it and load it like an sklearn dataset
    train_path = "../data/train_file.csv"
    train_dataset = ecommerce_sentiment_analysis(train_path, dev_percent)

    # 2. Vectorize the train_dataset
    tf_idf = vectorize(train_dataset)

    # 3. Split the train_dataset into train and validation

    # 3.1 First, ensure that the splits are of equal size
    split_factor = 5
    num_splits = train_dataset.embeddings.shape[0]
    while num_splits >= 10:
        num_splits = num_splits // split_factor

    # 3.2 Then, create the kfold object
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    # 4. Grid search with k-fold-cross-validation for the KNN classifier

    # 4.1. Define the parameter grid
    # NOTE 1 : constraints for k: k <= X_train.shape[0] and k % 2 == 1
    # NOTE 2 : to reduce search space, we increase the step size in the range
    # for k when dev_percent is 0
    # NOTE 3 : only metrics chosen are "cosine" and "euclidean"
    k_list = (
        list(range(1, 13, 2))
        if dev_percent == 0.001
        else list(range(1, 500, 16))
    )
    metrics_list = ["cosine", "euclidean"]
    params = {"n_neighbors": k_list, "metric": metrics_list}
    parameter_list = list(ParameterGrid(params))
    assert len(parameter_list) == len(k_list) * len(metrics_list)

    # 4.2. Perform grid search
    best, logs = grid_search_cv(parameter_list, train_dataset, kf)

    # 5. Plot the results
    _ = plot_logs(logs, metrics_list, best)

    # 6. Train the best model on the entire train_dataset
    #    and test it on the test_dataset

    # 6.1 Load test data
    test_path = "../data/test_file.csv"
    test_dataset = test_data(test_path, dev_percent)
    test_embeddings = tf_idf.transform(test_dataset.data)
    test_dataset.embeddings = test_embeddings

    X_train = train_dataset.embeddings
    y_train = train_dataset.target
    y_train = y_train.values
    X_test = test_dataset.embeddings

    print("X_train.shape")
    print(X_train.shape, "y_train.shape")
    print(y_train.shape, "X_test.shape", X_test.shape)
    # assert that X_train and X_test are not the same
    assert X_train is not X_test
    # assert that X_train and X_test have the same number of features
    assert X_train.shape[1] == X_test.shape[1]

    # 6.2 Train the best model on the entire train_dataset
    knn = KNNClassifier(n_neighbors=best.k, metric=best.metric)
    knn.fit(X_train, y_train)

    # 6.3 Test the best model on the test_dataset
    y_pred = knn.predict(X_test)
    # print("y_pred : ", y_pred)
    print("y_pred.shape : ", y_pred.shape)

    # 6.4 Save the predictions to a csv file
    df = pd.DataFrame(y_pred, columns=["target"])
    # if df["target"] is 1, then save it as "+1", else "-1" to match format file
    df["target"] = df["target"].apply(lambda x: "+1" if x == 1 else "-1")
    df.to_csv("predictions.csv", index=False, header=False)
