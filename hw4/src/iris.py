import umap

import numpy as np
import pandas as pd

from kmeans import KMeans
from load_iris import load_iris, to_numpy

if __name__ == "__main__":
    SEED = 42
    INIT_METHOD = "kmeans++"
    K = 3

    np.random.seed(seed=SEED)
    data = to_numpy(load_iris())

    reducer = umap.UMAP(
        init="spectral",
        metric="cosine",
        min_dist=0,
        n_components=2,
        n_neighbors=75,
        n_epochs=1000,
        random_state=SEED,
    )
    data = reducer.fit_transform(data)
    print(f"Data shape after UMAP: {data.shape}")

    kmeans = KMeans(k=K, init_method=INIT_METHOD)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    print(labels)
    with open(
        f"../outputs/{SEED}_{INIT_METHOD}_UMAP_density_kmeans_advanced_iris_labels.txt",
        "w",
    ) as f:
        for label in labels:
            f.write(f"{str(label)}\n")
