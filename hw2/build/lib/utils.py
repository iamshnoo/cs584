from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# set sns style
sns.set_style("darkgrid")
sns.set_palette("pastel")


def visualize_sparse(X: csr_matrix):

    temp = X.toarray()
    dense_size = np.array(temp).nbytes / 1e6
    sparse_size = (X.data.nbytes + X.indptr.nbytes + X.indices.nbytes) / 1e6
    print(f"Dense size: {dense_size:.2f} MB")
    print(f"Sparse size: {sparse_size:.2f} MB")

    # create 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    sns.barplot(x=["Dense", "Sparse"], y=[dense_size, sparse_size], ax=ax1)
    # label the bars with the size
    for i, v in enumerate([dense_size, sparse_size]):
        ax1.text(i, v, f"{str(round(v, 2))} MB", fontweight="bold", ha="center")
    ax1.set_title("Size of Dense vs Sparse Matrix representation of data")
    ax1.set_ylabel("MB")

    # plot the sparse matrix
    sns.heatmap(temp, ax=ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("Features (1 to 100,000)")
    ax2.set_ylabel("Patterns (0 to 799)")
    ax2.set_title("Sparse Matrix Representation of Data (Heatmap)")
    plt.savefig("../figs/sparse.png")
    plt.show()
