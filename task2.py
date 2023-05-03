from pandas import DataFrame as df
from typing import Tuple, Any

from task1 import preprocess
from knn import KNNClassifier, EUCLIDEAN, MANHATTAN, MINKOWSKI
from sk_tree import DecisionTreeClassifier

from math import sqrt


def load_data(
    test="./test-final.csv", train="test-final.csv", corr_threshold=0.75
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
    """
    returns ((train_X, train_y), (test_X, test_y))
    """
    # Load the data
    to_drop, df = preprocess(train, corr_threshold=corr_threshold)
    test_df = preprocess(test, to_drop=to_drop)  # Drop the same columns

    # Split the data into x and y
    train = df.iloc[:, :-1].values.tolist(), df.iloc[:, -1].values.tolist()

    # Split the data into x and y
    test = test_df.iloc[:, :-1].values.tolist(), test_df.iloc[:, -1].values.tolist()
    return train, test


def knn(train_X, train_y, test_X, test_y, n_neighbors=5, distance_measure=EUCLIDEAN):
    from sklearn.metrics import accuracy_score

    # Define classifiers
    knn = KNNClassifier(k=n_neighbors, distance_measure=distance_measure)

    # Fit the models
    knn.fit(train_X, train_y)

    # Compute the accuracy
    knn_pred = knn.estimate(test_X)
    return accuracy_score(test_y, knn_pred)


def knn_parameter_sweep(
    train_X, train_y, test_X, test_y, n_neighbors_min=5, n_neighbors_max=0
):
    """
    Sweeps over the parameters n_neighbors and distance_metric
    """
    if n_neighbors_min < 1:
        raise ValueError("n_neighbors_min must be greater than 0")
    if n_neighbors_max < n_neighbors_min and n_neighbors_max > 0:
        raise ValueError("n_neighbors_max must be greater than n_neighbors_min")
    if n_neighbors_max < 1:
        n_neighbors_max = sqrt(
            len(train_y)
        )  # sqrt(n) is a good heuristic for n_neighbors_max
    scores = [[0 for _ in range(n_neighbors_min, n_neighbors_max)] for _ in range(3)]
    measures = [EUCLIDEAN, MANHATTAN, MINKOWSKI]
    measures_str = ["euclidean", "manhattan", "minkowski"]
    for i, measure in enumerate(measures):
        for n_neighbors in range(n_neighbors_min, n_neighbors_max + 1):
            scores[i][n_neighbors - 1] = knn(
                train_X, train_y, test_X, test_y, n_neighbors, measure
            )
            print(
                f"n_neighbors: {n_neighbors}, metric: {measures_str[i]}, score: {scores[i][n_neighbors - 1]}"
            )

    import matplotlib.pyplot as plt

    plt.subplot(4, 4)
    plt.title("KNN parameter sweep")
    plt.xlabel("n_neighbors")
    plt.ylabel("accuracy")

    plt.plot(range(1, n_neighbors_max + 1), scores[0], label="euclidean")
    plt.plot(range(1, n_neighbors_max + 1), scores[1], label="manhattan")
    plt.plot(range(1, n_neighbors_max + 1), scores[2], label="minkowski")
    plt.legend()
    plt.show()


if __name__ == "__main__":

train, test = load_data()
knn_parameter_sweep(*train, *test)
