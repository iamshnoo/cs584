from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import csr_matrix

# set sns style
sns.set_style("darkgrid")
sns.set_palette("pastel")


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.dist = Counter(y)
        self.sparsity = self.X.nnz / (self.X.shape[0] * self.X.shape[1])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __repr__(self) -> str:
        return f"Dataset(X={self.X}, y={self.y})"


def create_sparse_from_locations(locations):
    # a simple function to map the data format in the question
    # to the format required by scipy.sparse.csr_matrix
    locations_dict = dict(enumerate(locations))
    for key, value in locations_dict.items():
        locations_dict[key] = [int(i.split(":")[0]) for i in value]
        locations_dict[key] = [(key, i) for i in locations_dict[key]]
    locations_list = [item for sublist in locations_dict.values() for item in sublist]
    row = [i[0] for i in locations_list]  # min=0, max=799
    col = [i[1] for i in locations_list]  # min=1, max=100,000
    assert len(row) == len(col)
    num_locations = len(row)
    data = [1] * num_locations
    # for each item in col, subtract 1 (to make it 0-indexed)
    col = [i - 1 for i in col]
    return row, col, data


# (0, 96) 1
# (0, 183) 1


def load_dataset(filename="../data/train.txt", mode="train"):

    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        if mode == "train":
            locations = [line[1:] for line in lines]
            # first element of each line is the class label
            class_labels = [int(line[0]) for line in lines]

        else:
            locations = lines
            # class labels is a dummy list of -1s for test data
            class_labels = [-1] * len(locations)

        # generate three lists corresponding to X[row, col] = data
        row, col, data = create_sparse_from_locations(locations)

        # create a csr sparse matrix from row, col, data
        X = csr_matrix((data, (row, col)), shape=(len(lines), max(col) + 1))
    return Dataset(X, class_labels)


def visualize_sparse(X):

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


if __name__ == "__main__":
    path = "../data/train.txt"
    mode = "train"
    dataset = load_dataset(path, mode)
    print(f"X shape: {dataset.X.shape}")
    print(f"Number of total samples: {len(dataset.y)}")
    print(f"Class distribution: {dataset.dist}")
    print(f"Sparsity: {dataset.sparsity}")
    # uncomment the following line to plot figs/sparse.png
    # visualize_sparse(dataset.X)
