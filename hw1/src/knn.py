import warnings
from copy import deepcopy

import numpy as np
import scipy.stats as stats
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin

from pre_process import ecommerce_sentiment_analysis
from vectorize import train_validation_split, vectorize

warnings.simplefilter(action="ignore", category=FutureWarning)

DEBUG = False


class KNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, metric="euclidean"):
        self.k = n_neighbors
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    # utility function to find k nearest neighbors (indices, distances)
    def _k_neighbors(self, pairwise_dist, calculate_distance=False):
        # since we are choosing k elements from each row of sorted of pairwise_dist,
        # the value of k needs to be less than the number of columns in pairwise_dist
        assert self.k <= self.X_train.shape[0]

        if DEBUG:
            calculate_distance = True
        # use stable sorting (based on timsort) instead of the default quicksort
        # to choose the lower index in case of ties in distance
        k_nearest_neighbors_indices = np.argsort(
            pairwise_dist, axis=1, kind="stable"
        )[:, : self.k]
        if not calculate_distance:
            return k_nearest_neighbors_indices
        sorted_pairwise_dist = np.sort(pairwise_dist, axis=1)
        k_nearest_neighbors_distances = sorted_pairwise_dist[:, : self.k]
        if DEBUG:
            for i in range(X_test.shape[0]):
                for j in range(X_train.shape[0]):
                    print("{:.5f}".format(pairwise_dist[i, j]), end=" ")
                print()
            print("-" * 80)
            sorted_pairwise_dist = np.sort(pairwise_dist, axis=1)
            for i in range(X_test.shape[0]):
                for j in range(X_train.shape[0]):
                    # print upto 3 decimal places
                    print("{:.3f}".format(sorted_pairwise_dist[i, j]), end=" ")
                print()
            self._print_helper(k_nearest_neighbors_distances)
            print(k_nearest_neighbors_indices)
            print("-" * 80)
            return k_nearest_neighbors_indices
        return k_nearest_neighbors_indices, k_nearest_neighbors_distances

    def predict(self, X_test):
        # [i, j]th element of pairwise_dist is the distance between X_test[i],
        # X_train[j] where i goes from 0 to X_test.shape[0] and j goes from 0 to
        # X_train.shape[0].
        # row i of pairwise_dist is the distance between X_test[i] and X_train.
        pairwise_dist = metrics.pairwise_distances(
            X_test, self.X_train, metric=self.metric
        )
        k_nearest_neighbors_indices = self._k_neighbors(pairwise_dist, False)
        # reshape y_ from (n,) to (n, 1) to make it compatible with k_nearest_neighbors_indices
        y_ = deepcopy(self.y_train)
        if DEBUG:
            self._print_helper(y_)
        y_ = y_.reshape((-1, 1))
        if DEBUG:
            self._print_helper(y_)
        mode, _ = stats.mode(
            y_[k_nearest_neighbors_indices, 0], axis=1, keepdims=False
        )
        if DEBUG:
            print("-" * 80)
            print(y_[k_nearest_neighbors_indices, 0])
            print(mode)
            print("-" * 80)

        y_pred = np.empty(
            (X_test.shape[0], y_.shape[1]), dtype=self.y_train[0].dtype
        )
        if DEBUG:
            self._print_helper(y_pred)
        y_pred[:, 0] = mode
        if DEBUG:
            self._print_helper(y_pred)
        y_pred = y_pred.ravel()
        if DEBUG:
            self._print_helper(y_pred)
        return y_pred

    # utility print function for debugging
    def _print_helper(self, arg0):
        print("-" * 80)
        print(arg0)
        print("-" * 80)

    # utility function to calculate accuracy
    def _accuracy(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / y_test.shape[0]

    # function to calculate accuracy on validation set
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return self._accuracy(y_test, y_pred)


if __name__ == "__main__":

    # 1. Read raw data, process it and load it like an sklearn dataset
    path = "../data/train_file.csv"
    dev_percent = 0.001
    dataset = ecommerce_sentiment_analysis(path, dev_percent)

    # 2. Vectorize the dataset
    tf_idf = vectorize(dataset)

    # 3. Split the dataset into train and validation
    X_train, X_test, y_train, y_test = train_validation_split(
        dataset.embeddings, dataset.target
    )

    y_train = y_train.values
    y_test = y_test.values

    # 4. Train a KNN classifier
    knn = KNNClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X_train, y_train)

    # 5. Evaluate the classifier on the validation set
    y_pred = knn.predict(X_test)
    print("y_pred : ", y_pred)
    print("y_test : ", y_test)
    print(y_test.dtype, type(y_test), y_pred.dtype, type(y_pred))
    # print("Accuracy : ", knn.score(X_test, y_test))
    print(metrics.classification_report(y_test, y_pred))
    print("-" * 80)
