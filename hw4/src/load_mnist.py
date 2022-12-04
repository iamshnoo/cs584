import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_mnist():

    with open("../data/mnist/test.txt") as f:
        lines = f.readlines()

    lines = [x.strip().split(" ") for x in lines]
    lines = np.array(lines)
    data = [lines[i][0].split(",") for i in range(len(lines))]
    data = [list(map(int, data[i])) for i in range(len(data))]
    data = np.array(data)
    print("Loaded MNIST data with shape: ", data.shape)
    return data


# loads a random image and displays it
def visualize_image_mnist(data):
    # choose a data index randomly
    idx = np.random.randint(0, data.shape[0])
    # reshape the data to 28x28
    img = data[idx].reshape(28, 28)
    # plot the image
    plt.imshow(img, cmap="gray")
    plt.show()


if __name__ == "__main__":
    data = load_mnist()
    visualize_image_mnist(data)
