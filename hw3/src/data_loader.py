from collections import Counter

from scipy.sparse import csr_matrix


class Dataset:
    def __init__(self, X: csr_matrix, y: list) -> None:
        self.X = X
        self.y = y
        self.dist = Counter(y)
        self.sparsity = self.X.nnz / (self.X.shape[0] * self.X.shape[1])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]

    def __repr__(self) -> str:
        return f"Dataset(X={self.X}, y={self.y})"


def create_sparse_from_locations(locations: list) -> tuple:
    # a simple function to map the data format given in the question
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


def load_dataset(filename: str = "../data/train.txt", mode: str = "train") -> Dataset:

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

    # return it by wrapping it in a Dataset object
    return Dataset(X, class_labels)


if __name__ == "__main__":
    path = "../data/train.txt"
    mode = "train"
    dataset = load_dataset(path, mode)
    print(f"X shape: {dataset.X.shape}")
    print(f"Number of total samples: {len(dataset.y)}")
    print(f"Class distribution: {dataset.dist}")
    print(f"Sparsity: {dataset.sparsity}")
