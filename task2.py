from pandas import DataFrame as df
from typing import Tuple, Any

from task1 import preprocess
from knn import KNNClassifier, EUCLIDEAN, MANHATTAN, MINKOWSKI
from sk_tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from math import sqrt

top_5 = lambda x, y: sorted(zip(x, y), key=lambda x: x[1], reverse=True)[:5]


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
    knn = KNNClassifier(k=n_neighbors, distance_measure=distance_measure)
    knn.fit(train_X, train_y)
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
        n_neighbors_max = int(sqrt(len(train_y)))

    scores = [[0 for _ in range(n_neighbors_min, n_neighbors_max)] for _ in range(3)]
    measures = [EUCLIDEAN, MANHATTAN, MINKOWSKI]
    measures_str = ["euclidean", "manhattan", "minkowski"]
    for i, measure in enumerate(measures):
        for index, n_neighbors in enumerate(range(n_neighbors_min, n_neighbors_max)):
            scores[i][index] = knn(
                train_X, train_y, test_X, test_y, n_neighbors, measure
            )
            print(
                f"n_neighbors: {n_neighbors}, metric: {measures_str[i]}, accuracy: {scores[i][index]}"
            )

    import matplotlib.pyplot as plt

    plt.title("KNN parameter sweep")
    plt.ylabel("accuracy")

    plt.plot(range(1, n_neighbors_max + 1), scores[0], label="euclidean")
    plt.plot(range(1, n_neighbors_max + 1), scores[1], label="manhattan")
    plt.plot(range(1, n_neighbors_max + 1), scores[2], label="minkowski")
    plt.legend()
    plt.show()


def tree(
    train_X,
    train_y,
    test_X,
    test_y,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
):
    # Define classifiers
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    # Fit the models
    tree.fit(train_X, train_y)

    # Compute the accuracy
    tree_pred = tree.predict(test_X)
    return accuracy_score(test_y, tree_pred)


def tree_parameter_sweep(
    train_X,
    train_y,
    test_X,
    test_y,
    depth_range=range(1, 10),
    min_samples_split_range=range(2, 10),
    min_samples_leaf_range=range(1, 10),
):
    """
    Sweeps over the parameters max_depth, min_samples_split and min_samples_leaf
    """
    mean = lambda x: sum(x) / len(x)
    scores_depth = [0 for _ in depth_range]
    scores_split = [0 for _ in min_samples_split_range]
    scores_leaf = [0 for _ in min_samples_leaf_range]
    for index, depth in enumerate(depth_range):
        scores_depth[index] = mean(
            [tree(train_X, train_y, test_X, test_y, max_depth=depth) for _ in range(10)]
        )
        print(f"max_depth: {depth}, score: {scores_depth[index]}")
    for index, split in enumerate(min_samples_split_range):
        scores_split[index] = mean(
            [
                tree(train_X, train_y, test_X, test_y, min_samples_split=split)
                for _ in range(10)
            ]
        )
        print(f"min_samples_split: {split}, score: {scores_split[index]}")
    for index, leaf in enumerate(min_samples_leaf_range):
        scores_leaf[index] = mean(
            [
                tree(train_X, train_y, test_X, test_y, min_samples_leaf=leaf)
                for _ in range(10)
            ]
        )
        print(f"min_samples_leaf: {leaf}, score: {scores_leaf[index]}")

    # Find top 5 scores
    top_depth = top_5(scores_depth, depth_range)
    top_split = top_5(scores_split, min_samples_split_range)
    top_leaf = top_5(scores_leaf, min_samples_leaf_range)

    print(f"Top 5 max_depth: {top_depth}")
    print(f"Top 5 min_samples_split: {top_split}")
    print(f"Top 5 min_samples_leaf: {top_leaf}")

    combined_scores = []
    # Rerun for all combinations of top 5
    for depth, _ in top_depth:
        for split, _ in top_split:
            for leaf, _ in top_leaf:
                combined_scores.append(
                    (
                        (depth, split, leaf),
                        tree(
                            train_X,
                            train_y,
                            test_X,
                            test_y,
                            max_depth=depth,
                            min_samples_split=split,
                            min_samples_leaf=leaf,
                        ),
                    )
                )

                print(
                    f"max_depth: {depth}, min_samples_split: {split}, min_samples_leaf: {leaf}, score: {combined_scores[-1]}"
                )
    best = sorted(combined_scores, key=lambda x: x[1], reverse=True)[0]
    print(f"Best combination: {best}")

    import matplotlib.pyplot as plt

    plt.title("Decision Tree parameter sweep")
    plt.ylabel("accuracy")

    plt.plot(depth_range, scores_depth, label="max_depth")
    plt.plot(min_samples_split_range, scores_split, label="min_samples_split")
    plt.plot(min_samples_leaf_range, scores_leaf, label="min_samples_leaf")
    plt.legend()
    plt.show()


def random_forest(
    train_X,
    train_y,
    test_X,
    test_y,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    n_estimators=10,
):
    # Define classifiers
    forest = RandomForestClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
    )

    # Fit the models
    forest.fit(train_X, train_y)

    # Compute the accuracy
    forest_pred = forest.predict(test_X)
    return accuracy_score(test_y, forest_pred)


def random_forest_parameter_sweep(
    train_X,
    train_y,
    test_X,
    test_y,
    depth_range=range(1, 10),
    min_samples_split_range=range(2, 10),
    n_estimators_range=range(1, 10),
):
    """
    Sweeps over the parameters max_depth, min_samples_split, and n_estimators
    """
    score_depth, score_split, score_est = [], [], []
    for depth in depth_range:
        score_depth.append(
            random_forest(train_X, train_y, test_X, test_y, max_depth=depth)
        )
        print(f"max_depth: {depth}, score: {score_depth[-1]}")
    for split in min_samples_split_range:
        score_split.append(
            random_forest(train_X, train_y, test_X, test_y, min_samples_split=split)
        )
        print(f"min_samples_split: {split}, score: {score_split[-1]}")
    for estimators in n_estimators_range:
        score_est.append(
            random_forest(train_X, train_y, test_X, test_y, n_estimators=estimators)
        )
        print(f"n_estimators: {estimators}, score: {score_est[-1]}")
    top_depth = top_5(depth_range, score_depth)
    top_split = top_5(min_samples_split_range, score_split)
    top_est = top_5(n_estimators_range, score_est)
    print(f"Top 5 max_depth: {top_depth}")
    print(f"Top 5 min_samples_split: {top_split}")
    print(f"Top 5 n_estimators: {top_est}")

    # Rerun for all combinations of top 5

    print("Checking all combinations of top 5...")
    scores = []
    for depth, _ in top_depth:
        for split, _ in top_split:
            for est, _ in top_est:
                scores.append(
                    (
                        (depth, split, est),
                        random_forest(
                            train_X,
                            train_y,
                            test_X,
                            test_y,
                            max_depth=depth,
                            min_samples_split=split,
                            n_estimators=est,
                        ),
                    )
                )

    def best(scores):
        eval = lambda x: x[1] / (x[0][0] * x[0][1] * x[0][2])
        return sorted(scores, key=eval, reverse=True)[0]

    best = best(scores)
    print(f"Best combination: {best}")

    import matplotlib.pyplot as plt

    plt.title("Random Forest parameter sweep")
    plt.ylabel("accuracy")

    plt.plot(depth_range, score_depth, label="max_depth")
    plt.plot(min_samples_split_range, score_split, label="min_samples_split")
    plt.plot(n_estimators_range, score_est, label="n_estimators")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train, test = load_data()
    random_forest_parameter_sweep(
        *train,
        *test,
        depth_range=range(1, 50),
        min_samples_split_range=range(2, 10),
        n_estimators_range=range(1, 100),
    )
