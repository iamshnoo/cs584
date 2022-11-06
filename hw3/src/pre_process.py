from collections import Counter

import pandas as pd
import yaml
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2

from data_loader import Dataset


def imbalance_class_sampling(dataset: Dataset, method: str = "SMOTETomek") -> None:
    with open("configs.yaml", "r") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    # apply oversampling + undersampling
    if method == "SMOTETomek":
        X_res, y_res = SMOTETomek(random_state=CONFIG["random_state"]).fit_resample(
            dataset.X, dataset.y
        )
    # apply oversampling
    elif method == "SMOTE":
        X_res, y_res = SMOTE(random_state=CONFIG["random_state"]).fit_resample(
            dataset.X, dataset.y
        )
    # apply undersampling
    elif method == "RandomUnderSampler":
        X_res, y_res = RandomUnderSampler(
            random_state=CONFIG["random_state"]
        ).fit_resample(dataset.X, dataset.y)
    # haven't tried other methods
    else:
        raise ValueError("Invalid method")
    dataset.X = X_res
    dataset.y = y_res
    dataset.dist = Counter(y_res)
    dataset.sparsity = dataset.X.nnz / (dataset.X.shape[0] * dataset.X.shape[1])


def step_1(dataset: Dataset, p: int = 0.99) -> VarianceThreshold:
    sel = VarianceThreshold(threshold=(p * (1 - p)))
    dataset.X = sel.fit_transform(dataset.X)
    return sel


def step_2(dataset: Dataset, k: int = 7000) -> SelectKBest:
    sel = SelectKBest(chi2, k=k)
    dataset.X = sel.fit_transform(dataset.X, dataset.y)
    return sel


def step_3a(dataset: Dataset, n_trees: int = 10000, n: int = 750) -> pd.Series:
    with open("configs.yaml", "r") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    feature_names = [f"feature {i}" for i in range(dataset.X.shape[1])]
    forest = BalancedRandomForestClassifier(
        random_state=CONFIG["random_state"], n_estimators=n_trees, n_jobs=-1
    )
    forest.fit(dataset.X, dataset.y)
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    indices = [int(elem.split(" ")[1]) for elem in forest_importances.nlargest(n).index]
    dataset.X = dataset.X[:, indices]
    return forest_importances


def step_3b(dataset: Dataset, forest_importances: pd.Series, n: int = 750) -> None:
    idxs = [int(elem.split(" ")[1]) for elem in forest_importances.nlargest(n).index]
    dataset.X = dataset.X[:, idxs]


class PreProcess:
    def __init__(
        self,
        p: int = 0.99,
        k: int = 7000,
        num_trees: int = 10000,
        num_ftrs_final: int = 750,
    ) -> None:
        self.var_thresh = None
        self.kbest = None
        self.brf_importances = None
        self.p = p
        self.k = k
        self.num_trees = num_trees
        self.num_ftrs_final = num_ftrs_final

    # fit on train set
    def fit(self, dataset: Dataset):
        self.var_thresh = step_1(dataset, self.p)
        self.kbest = step_2(dataset, self.k)
        self.brf_importances = step_3a(dataset, self.num_trees, self.num_ftrs_final)

    # transform only on test set
    def transform(self, dataset: Dataset):
        dataset.X = self.var_thresh.transform(dataset.X)
        dataset.X = self.kbest.transform(dataset.X)
        step_3b(dataset, self.brf_importances, self.num_ftrs_final)
