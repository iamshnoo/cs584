import numpy as np
import pandas as pd
from load_iris import load_iris, to_numpy, to_df
from sklearn.metrics import silhouette_score, davies_bouldin_score


class KMeans:
    def __init__(self, k, max_iter=500, init_method="kmeans++"):
        self.k = k
        self.max_iter = max_iter
        self.inertia = np.inf
        self.init_method = init_method

    def init_centroids(self, X, num_centroids=None):
        if self.init_method == "forgy":
            self.init_forgy(X, num_centroids)
        elif self.init_method == "kmeans++":
            self.init_kmeans(X, num_centroids)

    def init_forgy(self, X, num_centroids=None):
        # "Forgy" initialization method
        if num_centroids is None:
            num_centroids = self.k
        assert (
            num_centroids <= X.shape[0]
        ), "Number of centroids must be less than number of data points"
        rand_indices = np.random.choice(X.shape[0], num_centroids, replace=False)
        self.centroids = X[rand_indices]
        print(self.centroids)

    def init_kmeans(self, X, num_centroids=None):
        # "kmeans++" initialization method
        if num_centroids is None:
            num_centroids = self.k
        assert (
            num_centroids <= X.shape[0]
        ), "Number of centroids must be less than number of data points"
        self.centroids = np.zeros((num_centroids, X.shape[1]))
        self.centroids[0] = X[np.random.choice(X.shape[0], 1, replace=False)]
        for i in range(1, num_centroids):
            dists = np.linalg.norm(X - self.centroids[i - 1], axis=1)
            probs = dists / np.sum(dists)
            self.centroids[i] = X[
                np.random.choice(X.shape[0], 1, replace=False, p=probs)
            ]

    def calc_inertia(self, X):
        return sum(
            np.linalg.norm(X[i] - self.centroids[int(self.labels[i])]) ** 2
            for i in range(X.shape[0])
        )

    def fit(self, X):
        self.init_centroids(X)
        self.labels = np.zeros(X.shape[0])
        self.iter = 0
        print("-" * 80)
        print("Initial inertia: ", self.calc_inertia(X))

        # Convergence criteria 1: Number of iterations is less than max_iter
        while self.iter < self.max_iter:

            # Assignment step for each data point
            for i in range(X.shape[0]):
                self.labels[i] = np.argmin(
                    np.linalg.norm(X[i] - self.centroids, axis=1)
                )

            # Update step for each centroid
            for i in range(self.k):
                self.centroids[i] = np.mean(X[self.labels == i], axis=0)

            self.iter += 1

            # Convergence criteria 2: Inertia should be less than previous inertia
            if self.calc_inertia(X) >= self.inertia:
                break

            self.inertia = self.calc_inertia(X)
            self.metric1 = davies_bouldin_score(X, self.labels)
            self.metric2 = silhouette_score(X, self.labels)
            print("-" * 80)
            print("Iteration: ", self.iter)
            print("Inertia: ", self.inertia)
            print("Sklearn metrics : ")
            print("Silhouette score: ", self.metric2)
            print("Davies-Bouldin score: ", self.metric1)

        # Calculate inertia and homogeneous after convergence
        self.inertia = self.calc_inertia(X)
        self.metric1 = davies_bouldin_score(X, self.labels)
        self.metric2 = silhouette_score(X, self.labels)
        print("-" * 80)
        print("Final inertia: ", self.inertia)
        print("Sklearn metrics : ")
        print("Silhouette score: ", self.metric2)
        print("Davies-Bouldin score: ", self.metric1)
        print("-" * 80)

    def predict(self, X):
        # return the labels for each data point in X
        labels = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            labels[i] = np.argmin(np.linalg.norm(X[i] - self.centroids, axis=1))
        labels = [int(label + 1) for label in labels]
        return labels
