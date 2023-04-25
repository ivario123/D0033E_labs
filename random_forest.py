# Defines a simple random forest classifier

from math import floor
from typing import List, Tuple

import numpy as np
from scipy.stats import entropy

from tree import TreeClassifier


class RandomForest:
    def __init__(
        self,
        n_trees: int = 10,
        max_d: int = 5,
        min_samples_split: int = 2,
        bag_size=0.2,
    ) -> None:
        self.n_trees = n_trees
        self.max_d = max_d
        self.min_samples_split = min_samples_split
        self.trees = []
        self.bag_size = bag_size

    def bootstrap(self, X: List[List], y: List) -> Tuple[List[List], List]:
        """
        Creates a bootstrap sample of X and y
        """
        n = len(X)
        indices = np.random.choice(n, floor(n * self.bag_size), replace=True)
        X_temp = np.array(X)
        y_temp = np.array(y)
        X = X_temp[indices].tolist()
        y = y_temp[indices].tolist()

        # print(f"Bootstrapping with {X},{y} samples")
        return X, y

    def fit(self, x_in: List[List], y_in: List) -> None:
        bags_X = []
        bags_y = []
        for _ in range(self.n_trees):
            bag_X, bag_y = self.bootstrap(x_in, y_in)
            bags_X.append(bag_X)
            bags_y.append(bag_y)

        def validate_tree(tree: TreeClassifier, X: List[List], y: List) -> bool:
            """
            Validates the tree
            """
            miss_count = 0
            for i in range(len(X)):
                est = tree.estimate(X[i])
                if est != y[i]:
                    miss_count += 1
            print(f"\rMiss count: {miss_count} criteria: < {0.25 * len(X)}")
            return miss_count < 0.25 * len(X)

        def fit_tree(X: List[List], y: List, count: int) -> TreeClassifier:
            print(f"\r{'='*20} Building tree {count} {'='*20}")
            self.progress(count)
            tree = TreeClassifier(self.max_d, self.min_samples_split)
            tree.fit(X, y)
            print(f"\rValidating tree {count}")
            self.progress(count)
            if not validate_tree(tree, X, y):
                raise Exception("Tree validation failed")
            return tree

        self.trees = [
            fit_tree(X, y, id) for id, (X, y) in enumerate(zip(bags_X, bags_y))
        ]

    def progress(self, count):
        percent = floor(100 * count / self.n_trees)
        print(f"[{'='*(percent//10)+'>'+(10-percent//10)*'.'}] {percent}%", end="\r")

    def estimate(self, X: List) -> List:
        """
        Predicts a single instance
        """
        predictions = []
        if type(X) == float:
            print(f"Predicting with {X}")
            assert False
        for tree in self.trees:
            predictions.append(tree.estimate(X))
        print(f"Predictions: {predictions}")
        # Find the highest voted class
        votes = [0 for _ in range(0, floor(max(predictions)) + 1)]
        for prediction in predictions:
            print(prediction)
            votes[floor(prediction)] += 1
        print(f"Votes: {len(votes)} Most votes on one item: {max(votes)}")
        m = [i for i, x in enumerate(votes) if x == max(votes)][0]
        print(f"Most likely class: {m}")
        return m

    def score(self, X: List[List], y: List) -> float:
        """
        Computes the accuracy of the model
        """
        y_pred = self.estimate(X)
        return np.mean(np.array(y_pred) == np.array(y))

    def __str__(self) -> str:
        return f"RandomForest({self.n_trees}, {self.max_d}, {self.min_samples_split})"


from random import randint


def accuracy(forest: RandomForest, X: List[List], y: List) -> float:
    """
    Computes the accuracy of the given tree
    """
    correct = 0
    for i in range(0, len(X)):
        if type(X[i]) == float:
            print(X[i])
            assert False
        est = forest.estimate(X[i])
        print(f"Estimate: {est}, Actual: {y[i]}")
        if est == y[i]:
            correct += 1

    return correct / len(X)


from pandas import read_csv

# data = read_csv("data_processed.csv")
# Read data from the csv using pandas
df = read_csv("data_processed.csv")

# We don't need the text version of the label
# df = df.drop("gesture label", axis=1)


# Replace missing values with mean of that column
df = df.fillna(df.mean())
data = df.values.tolist()
labels = [row.pop() for row in data]  # Remove the label from the data
data = data[:]  # Only use the first 100 rows due to performance issues

data = [[float(x) for x in row] for row in data]  # Convert to floats
forest = RandomForest(max_d=10, n_trees=20, bag_size=0.5)
forest.fit(data, labels)

print(accuracy(forest, data, labels))
