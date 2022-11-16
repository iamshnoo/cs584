import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


class AdaBoost:
    def __init__(self, k=3, stump="decision_tree"):
        # k is the number of boosting rounds / number of weak learners
        self.k = k
        self.stump = stump
        self.models = []
        self.alphas = []
        # create a dictionary to store the models and alphas
        # in the format k : ['model_1': model, 'alpha_1': alpha]
        self.learners = {}

    def fit(self, X, y):
        y = np.multiply(y, 2) - 1
        # assert that y is binary, as Adaboost  expects y to be [-1,1]
        # and our data is [0,1]
        assert np.all(np.unique(y) == np.array([-1, 1]))

        # Step 1 : Initialize weights
        N, _ = X.shape
        w = np.full(N, (1 / N))

        # Step 2 : Iterate through k boosting rounds
        for i in range(self.k):
            # Step 3 : Train a weak learner on weighted data (sampling with
            # replacement according to w from the original dataset)
            if self.stump == "decision_tree":
                model = DecisionTreeClassifier(max_depth=15)
            elif self.stump == "logistic_regression":
                model = LogisticRegression()
            elif self.stump == "random_forest":
                model = RandomForestClassifier()
            elif self.stump == "svm":
                model = LinearSVC()
            else:
                raise ValueError("Invalid stump type")
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)
            # Step 4 : Calculate weighted error
            err = w.dot(y_pred != y)
            # Step 5 : if error > 0.5, then reset the weights and redo steps 3 and 4.
            if err > 0.5:
                w = np.full(N, (1 / N))
                model.fit(X, y, sample_weight=w)
                y_pred = model.predict(X)
                err = w.dot(y_pred != y)

            # Step 6 : Calculate alpha, account for numerical instability
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))

            # Step 7: Update weights
            w *= np.exp(-alpha * y * y_pred)
            z = np.sum(w)
            w /= z

            # Step 8: Store alpha and fitted weak learner
            self.learners[i] = list({"model": model, "alpha": alpha}.values())

    def predict(self, X):
        N, _ = X.shape
        y_pred = np.zeros(N)
        # Final prediction is the sum of the predictions of all the weak learners
        # weighted by their alphas
        for _, value in self.learners.items():
            model = value[0]
            alpha = value[1]
            y_pred += alpha * model.predict(X)
        return np.multiply((np.sign(y_pred) + 1), 0.5).astype(int)

    def get_params(self, deep=True):
        return {"k": self.k, "stump": self.stump}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
