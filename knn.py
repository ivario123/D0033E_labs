"""
Defines a simple k-nearest neighbors classifier

"""
import numpy as np
from typing import List, Tuple
from scipy.stats import entropy
from math import sqrt
from typing import Callable

EUCLIDEAN = lambda x1, x2, p: sqrt(sum([(x - y) ** 2 for x, y in zip(x1, x2)]))
MANHATTAN = lambda x1, x2, p: sum([abs(x - y) for x, y in zip(x1, x2)])
MINKOWSKI = lambda x1, x2, p: MANHATTAN(x1, x2) ** (1 / p)


class KNNClassifier:
    def __init__(
        self, distance_measure: Callable = EUCLIDEAN, k: int = 5, p: int = 2
    ) -> None:
        self.k = k
        self.distance_measure = distance_measure
        self.p = p
        self.X = None
        self.y = None
        self.labels = None

    def fit(self, X: List[List], y: List) -> None:
        """
        Fits the knn to the given data
        """
        self.X = X
        self.y = y
        self.labels = list(set(y))

    def euclidean(x1: List, x2: List, _: int) -> float:
        return sqrt(sum([(x - y) ** 2 for x, y in zip(x1, x2)]))

    def manhattan(x1: List, x2: List, _: int) -> float:
        return sum([abs(x - y) for x, y in zip(x1, x2)])

    def minkowski(x1: List, x2: List, p: int) -> float:
        return KNNClassifier.manhattan(x1, x2) ** (1 / p)

    def distance(self, x1: List, x2: List, method=euclidean) -> float:
        """
        This function assumes that x1:List of type that implements __add__(el:type(x2)) -> type(x1) and __sub__(el:type(x2)) -> type(x1)
        """
        if type(x1) != list:
            x1 = [x1]
        if type(x2) != list:
            x2 = [x2]

        return method(x1, x2, self.p)

    def estimate(self, x: List[any]) -> any:
        """
        Estimates the value of the given vector
        """
        if type(x[0]) == list:
            return [self.estimate(x_i) for x_i in x]

        distances = []
        for i, x_i in enumerate(self.X):
            distances.append((self.distance(x_i, x), self.y[i]))
        distances.sort(key=lambda x: x[0])
        votes = distances[: self.k]
        vote_map = [0 for _ in self.labels]
        for vote in votes:
            vote_map[self.labels.index(vote[1])] += 1
        return self.labels[vote_map.index(max(vote_map))]

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from task1 import preprocess

    # Define classifiers
    knn = KNNClassifier(k=5, distance_measure=MANHATTAN)

    # Load the data
    to_drop, df = preprocess("./train-final.csv", corr_threshold=0.75)
    test_df = preprocess("./test-final.csv", to_drop=to_drop)  # Drop the same columns

    # Split the data into x and y
    train_X = df.iloc[:, :-1].values.tolist()
    train_y = df.iloc[:, -1].values.tolist()

    # Split the data into x and y
    test_X = test_df.iloc[:, :-1].values.tolist()
    test_y = test_df.iloc[:, -1].values.tolist()

    # Fit the models
    knn.fit(train_X, train_y)

    # Compute the accuracy
    knn_pred = knn.estimate(test_X)
    print(f"KNN accuracy: {accuracy_score(test_y, knn_pred)}")
