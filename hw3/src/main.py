import seaborn as sns
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

from adaboost import AdaBoost
from data_loader import load_dataset
from pre_process import PreProcess, imbalance_class_sampling
from utils import cv_loop, cv_plot, print_shapes, test_predictions, print_test_set_dist

# set sns style
sns.set_style("darkgrid")
sns.set_palette("pastel")


if __name__ == "__main__":

    with open("configs.yaml", "r") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    train_dataset = load_dataset(CONFIG["train_path"], "train")
    test_dataset = load_dataset(CONFIG["test_path"], "test")
    imbalance_class_sampling(train_dataset, CONFIG["imbalance_method"])
    pre_process = PreProcess(
        p=CONFIG["p"],
        k=CONFIG["k"],
        num_trees=CONFIG["num_trees"],
        num_ftrs_final=CONFIG["num_ftrs_final"],
    )
    pre_process.fit(train_dataset)
    pre_process.transform(test_dataset)

    print_shapes(train_dataset, test_dataset)

    # Decision stump can be one of the following :
    # ["decision_tree", "logistic_regression", "linear_svm", "rbf_svm", "knn", "mlp"]
    clf = AdaBoost(k=1, stump=CONFIG["stump"])
    print(clf.k, clf.stump)
    cv = RepeatedStratifiedKFold(
        n_splits=CONFIG["cross_val_n_splits"],
        n_repeats=CONFIG["cross_val_n_repeats"],
        random_state=CONFIG["random_state"],
    )
    (
        train_mean_scores,
        test_mean_scores,
        train_std_scores,
        test_std_scores,
        best_k,
    ) = cv_loop(
        CONFIG["stump"],
        CONFIG["num_learners"],
        CONFIG["cross_val_n_splits"],
        CONFIG["cross_val_n_repeats"],
        CONFIG["metric"],
        train_dataset,
        clf,
        cv,
    )

    cv_plot(
        CONFIG["stump"],
        CONFIG["cv_f1_plot_path"],
        CONFIG["num_learners"],
        CONFIG["metric"],
        train_mean_scores,
        test_mean_scores,
        train_std_scores,
        test_std_scores,
    )

    y_pred = test_predictions(
        CONFIG["stump"], CONFIG["final_predictions_path"], train_dataset, test_dataset, clf, best_k
    )

    print("Best k = ", best_k)
    print("-" * 80)
    print_test_set_dist(test_dataset, y_pred)
