import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# set sns style
sns.set_style("darkgrid")
sns.set_palette("pastel")


def data_analysis():

    with open("../data/train.txt", "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        class_labels = [int(line[0]) for line in lines]
        num_features = [len(line[1:]) for line in lines]
        max_feature_index = [
            max(int(feature.split(":")[0]) for feature in line[1:]) for line in lines
        ]

        print("Number of patterns: ", len(lines))
        smallest_line = 9999
        largest_line = 0
        lengths = []
        print("-" * 80)
        for line in lines:
            line_length = len(line)
            if line_length > 2000:
                if int(line[0]) == 0:
                    print(
                        f"[Outlier(0)] Class label = {line[0]}, number of features = {line_length}"
                    )
                else:
                    print(
                        f"[Outlier(1)] Class label = {line[0]}, number of features = {line_length}"
                    )
            lengths.append(line_length)
            smallest_line = min(smallest_line, len(line[1:]))
            largest_line = max(largest_line, len(line[1:]))
        print("-" * 80)
        print("Smallest line: ", smallest_line)
        print("Largest line: ", largest_line)

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(8, 8)

        sns.histplot(
            x=num_features,
            hue=class_labels,
            ax=axs[0, 0],
            kde=True,
            bins=50,
            edgecolor=".5",
        )

        mu0, sigma0 = norm.fit(num_features)
        axs[0, 0].set_title(
            r"Number of features: $\mu=%.2f$, $\sigma=%.2f$" % (mu0, sigma0)
        )
        axs[0, 0].set_xlabel("Number of features")
        axs[0, 0].set_ylabel("Frequency")

        sns.barplot(
            x=[0, 1],
            y=[class_labels.count(0), class_labels.count(1)],
            ax=axs[0, 1],
            edgecolor=".5",
        )
        axs[0, 1].set_title("Distribution of class labels")
        axs[0, 1].set_xlabel("Class label")
        axs[0, 1].set_ylabel("Frequency")

        sns.scatterplot(
            x=num_features,
            y=class_labels,
            hue=class_labels,
            ax=axs[1, 0],
            edgecolor=".5",
        )
        axs[1, 0].set_yticks([0, 1])
        axs[1, 0].set_title("Number of features vs class labels")
        axs[1, 0].set_ylabel("Class labels")
        axs[1, 0].set_xlabel("Number of features")

        # sns plot max feature index
        sns.histplot(
            x=max_feature_index,
            hue=class_labels,
            ax=axs[1, 1],
            kde=True,
            bins=50,
            edgecolor=".5",
        )
        mu1, sigma1 = norm.fit(max_feature_index)
        axs[1, 1].set_title(
            r"Max feature index: $\mu=%.2f$, $\sigma=%.2f$" % (mu1, sigma1)
        )
        axs[1, 1].set_xlabel("Max feature index")
        axs[1, 1].set_ylabel("Frequency")

        fig.tight_layout()
        plt.savefig("../figs/eda.png", dpi=400)
        plt.show()


if __name__ == "__main__":
    data_analysis()
