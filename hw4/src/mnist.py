import umap

import numpy as np
import pandas as pd

from kmeans import KMeans
from load_mnist import load_mnist

if __name__ == "__main__":
    SEED = 42
    INIT_METHOD = "kmeans++"
    K = 10

    np.random.seed(seed=SEED)
    data = load_mnist()
    data = data.astype(np.float32) / 255.0

    reducer = umap.UMAP(
        init="spectral",
        metric="cosine",
        min_dist=0,
        n_components=2,
        n_neighbors=100,
        n_epochs=1000,
        random_state=SEED,
    )
    data = reducer.fit_transform(data)
    print(f"Data shape after UMAP: {data.shape}")

    kmeans = KMeans(k=K, init_method=INIT_METHOD)
    kmeans.fit(data)
    print(kmeans.centroids)
    labels = kmeans.predict(data)
    with open(f"../outputs/{SEED}_UMAP_mnist_advanced_labels.txt", "w") as f:
        for label in labels:
            f.write(f"{str(label)}\n")
