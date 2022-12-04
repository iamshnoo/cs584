import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# set sns style
sns.set_style("darkgrid")
sns.set_palette("pastel")

from kmeans import KMeans
from load_mnist import load_mnist

if __name__ == "__main__":
    SEED = 42
    INIT_METHOD = "kmeans++"

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

    k_vals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    final_inertias = []
    for k_val in k_vals:
        kmeans = KMeans(k=k_val, init_method=INIT_METHOD)
        kmeans.fit(data)
        print(kmeans.centroids)
        labels = kmeans.predict(data)
        final_inertias.append(kmeans.inertia)

    # plot the elbow curve using seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=k_vals, y=final_inertias, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Curve")
    plt.savefig("../figs/mnist_elbow_curve.png")
    plt.show()
