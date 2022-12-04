import numpy as np
import pandas as pd


def load_iris():
    with open("../data/iris/test.txt") as f:
        lines = f.readlines()
    lines = [x.strip().split(" ") for x in lines]
    data = [list(map(float, lines[i])) for i in range(len(lines))]
    data = pd.DataFrame(
        data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    return data


def to_numpy(data):
    return data.to_numpy()


def to_df(data):
    return pd.DataFrame(data)


if __name__ == "__main__":
    data = load_iris()
    print(data)
