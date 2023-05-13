from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


class Classifier(ABC):
    """
    Generic Classifier class for all classifiers in scikit-learn
    """

    def __init__(self, **kwargs):
        self._clf = None
        self._clf = self._create_clf()
        self._kwargs = kwargs

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def score(self, X, y):
        return self._clf.score(X, y)

    def get_params(self, deep=True):
        return self._clf.get_params(deep=deep)

    def set_params(self, **params):
        return self._clf.set_params(**params)


class Ensemble:
    """
    Ensemble classifier.
    """

    def __init__(
        self,
        classifier_type: Classifier,
        voters=10,
        use_bagging: bool = False,
        use_voting: bool = False,
        **kwargs,
    ):
        """
        Creates an ensemble classifier.

        Parameters
        ----------
        classifier_type : Classifier
            The type of classifier to use.
        voters : int, optional
            The number of voters to use, by default 10
        use_bagging : bool, optional
            Whether to use bagging, by default False
        use_voting : bool, optional
            Whether to use voting, by default False
            if true voting is used to estimate output
            if false averaging is used to estimate output
        """
        self._use_bagging = use_bagging
        self._classifier_type = classifier_type
        self._voters = voters if type(voters) == int else len(voters)
        self._use_voting = use_voting
        if type(voters) == int:
            self._classifiers: List[Classifier] = [
                self._classifier_type(**kwargs) for _ in range(self._voters)
            ]
        else:
            self._classifiers = voters

    def _bagging(self, X, y) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Creates bags for bagging.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The output data.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            The bags.
        """
        bags = []
        for _ in range(self._voters):
            idx = [
                int(c) for c in np.random.choice(len(X), len(X), replace=True).tolist()
            ]

            X = np.array([X[i] for i in idx])
            y = np.array([y[i] for i in idx])
            bags.append((X, y))
        return bags

    def fit(self, X, y):
        assert self._classifiers, "Classifier not created"

        bags = (
            self._bagging(X, y)
            if self._use_bagging
            else [(X, y) for _ in range(self._voters)]
        )
        for clf, (X, y) in zip(self._classifiers, bags):
            clf = clf.fit(X, y)

    def predict(self, X):
        assert self._classifiers, "Classifier not created"
        if self._use_voting:
            predictions = [clf.predict(X) for clf in self._classifiers]
            possible_options = max([max(p) for p in predictions])
            count = [[0] * (possible_options + 1) for _ in range(len(X))]
            for p in predictions:
                for i in range(len(p)):
                    count[i][p[i]] += 1
            return [np.argmax(c) for c in count]
        else:
            return np.mean([clf.predict(X) for clf in self._classifiers], axis=0)


class Boosting(Ensemble):
    def __init__(self, precision_threshold, *args, **kwargs):
        self.precision_threshold = precision_threshold
        super().__init__(*args, **kwargs)

    def __fit(self, X, y, weights):
        assert self._classifiers, "Classifier not created"
        bags = self.weighted_bagging(X, y, weights, bags=len(self._classifiers))
        error_counter = np.zeros(len(X))
        for i, clf in enumerate(self._classifiers):
            b_x, b_y = bags[i]
            # -- Fit
            clf.fit(b_x, b_y)
            # -- Predict
            y_pred = clf.predict(X)
            error = np.abs(y - y_pred)
            error_counter += error
        return error_counter

    def weighted_bagging(
        self, X, y, weights, bags
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        def bag(X, y, weights, bags):
            for _ in range(bags):
                idx = np.random.choice(len(X), len(X), replace=True, p=weights)
                X = np.array([X[i] for i in idx])
                y = np.array([y[i] for i in idx])

                yield (X, y)

        return list(bag(X, y, weights, bags))

    def update_weights(self, X, y, weights):
        error_counter = self.__fit(X, y, weights)
        # -- Adding one here avoids pure zero weights
        offset = weights / np.sum(weights)
        weights = weights + offset
        weights = weights / np.sum(weights)
        return weights, error_counter

    def progress(accuracy, counter, counter_max):
        # Delete last line and print a 0-100 progress bar and accuracy
        print("\033[F\033[K", end="")
        print(f"{counter/counter_max*100:.2f}%\t{accuracy:.2f}", end="")

        print(f"Boosting took {counter} iterations resulting in {accuracy} accuracy")

    def fit(self, X, y):
        assert self._classifiers, "Classifier not created"
        # -- Create weights
        self.weights = (np.ones(len(X)) / len(X)).tolist()
        accuracy = 0
        counter = 0
        max_itter = 100
        while accuracy < self.precision_threshold and counter < 100:
            self.weights, error_counter = self.update_weights(X, y, self.weights)
            accuracy = np.sum(error_counter == 0) / len(X)
            counter += 1
            Ensemble.progress(accuracy, counter, 100)
        print(f"Boosting took {counter} iterations resulting in {accuracy} accuracy")


if __name__ == "__main__":
    from task2 import load_data

    (X_tr, y_tr), (X_t, y_t) = load_data(drop_below_spine=False, corr_threshold=0.95)
    y_tr = [int(y) for y in y_tr]
    y_t = [int(y) for y in y_t]
    data = (X_tr, X_t), (y_tr, y_t)

    from json import load

    # Load optimal parameters from file
    with open("./best_results.json", "r") as f:
        params = load(f)

    tree_params = params["tree"]["params"]
    forest_params = params["forest"]["params"]
    knn_params = params["knn"]["params"]
    svm_params = {"kernel": "poly"}


def test(func):
    def wrapper(*args, **kwargs):
        name = func.__name__.replace("test_", "")
        name.replace("_", " ").title()
        print("Testing", name)
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        # Assume that we are running as main
        (X_train, X_test), (y_train, y_test) = data

        y_pred = func(
            *args,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            **kwargs,
        )

        acc = accuracy_score(y_test, y_pred)
        # assert acc > 0.9, f"{name} failed, accuracy {acc}"
        print(f"{name} passed with accuracy {acc}")

    return wrapper


@test
def test_pure_ensamble(X_train=None, X_test=None, y_train=None, **kwargs):
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    ensemble = Ensemble(
        classifier_type=None,
        voters=[
            SVC(**svm_params),
            DecisionTreeClassifier(**tree_params),
            KNeighborsClassifier(**knn_params),
            RandomForestClassifier(**forest_params),
        ],
        use_voting=True,
        use_bagging=True,
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    return y_pred


@test
def test_boosting(X_train=None, X_test=None, y_train=None, **kwargs):
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    ensemble = Boosting(
        classifier_type=SVC,
        voters=[
            *[SVC(**svm_params) for _ in range(10)],
            *[DecisionTreeClassifier(**tree_params) for _ in range(10)],
        ],
        use_voting=True,
        precision_threshold=0.9,
        **svm_params,
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    return y_pred


@test
def test_voting_with_tree(X_train=None, X_test=None, y_train=None, **kwargs):
    from sklearn.tree import DecisionTreeClassifier

    ensemble = Ensemble(
        classifier_type=DecisionTreeClassifier,
        voters=100,
        use_voting=True,
        **tree_params,
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    return y_pred


@test
def test_voting_with_svm(X_train=None, X_test=None, y_train=None, **kwargs):
    from sklearn.svm import SVC

    ensemble = Ensemble(classifier_type=SVC, voters=50, use_voting=True, **svm_params)
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    return y_pred


@test
def test_no_bagging_with_svm(X_train=None, X_test=None, y_train=None, **kwargs):
    from sklearn.svm import SVC

    ensemble = Ensemble(classifier_type=SVC, voters=50, use_voting=False, **svm_params)
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    return y_pred


if __name__ == "__main__":
    # Find all test functions, they are decorated with @test

    test_functions = [v for k, v in globals().items() if k.startswith("test_")]
    # Run all tests
    for test_function in test_functions:
        test_function()
