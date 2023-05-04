from math import sqrt
from typing import Any, Callable, Tuple, List

from pandas import DataFrame as df
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from knn import EUCLIDEAN, MANHATTAN, MINKOWSKI, KNNClassifier
from sk_tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from task1 import preprocess

from threading import Thread


def info(func: Callable):
    offset = lambda x, y: y // 2 - x // 2 if y > x else 0
    l = 50
    name = func.__name__.replace("_", " ").upper()

    def wrapper(*args, **kwargs):
        print("=" * l)

        print(f"{' '*offset(len(name),l)}{name}")
        print("=" * l)
        ret = func(*args, **kwargs)
        print("-" * l)
        print(f"{' '*(offset(len(str(ret)),l))}{ret}")
        print("-" * l)

    return wrapper


def itter(
    train_X,
    train_y,
    test_X,
    test_y,
    r: range,
    arg: str = "",
    target_func: Callable = None,
):
    assert target_func is not None, "target_func must be defined"
    assert arg != "", "arg must be defined"
    assert r is not None, "r must be defined"
    for i, val in enumerate(r):
        score = target_func(train_X, train_y, test_X, test_y, **{arg: val})
        yield i, score


top = lambda x, y: sorted(zip(x, y), key=lambda x: x[1], reverse=True)[:3]


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


def start_and_wait(threads: List[Thread]):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def knn(train_X, train_y, test_X, test_y, n_neighbors=5, distance_measure=EUCLIDEAN):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=distance_measure)
    knn.fit(train_X, train_y)
    knn_pred = knn.predict(test_X)
    return accuracy_score(test_y, knn_pred)


@info
def knn_parameter_sweep(
    train_X,
    train_y,
    test_X,
    test_y,
    scores,
    best,
    k_range=range(5, 20),
):
    """
    Sweeps over the parameters n_neighbors and distance_metric
    """
    measures_str = ["euclidean", "manhattan", "minkowski"]

    def itter_int(r: enumerate, scores: List):
        for i, measure in enumerate(measures_str):
            for index, k_range in r:
                scores[i][index] = knn(
                    train_X, train_y, test_X, test_y, k_range, measure
                )

    number_of_threads = 3
    enum = enumerate(k_range)

    def split_enum(e, sections) -> list[enumerate]:
        ret = [[] for _ in range(sections)]
        for i, val in e:
            ret[i % sections].append((i, val))
        return ret

    enums = split_enum(enum, number_of_threads)
    start_and_wait(
        [
            Thread(target=itter_int, args=(enums[i], scores))
            for i in range(number_of_threads)
        ]
    )

    # Find top 5 scores
    def optimal(scores):
        opt = (0, 0, float("-inf"))
        for index, el in enumerate(scores):
            for i, score in enumerate(el):
                if score > opt[2]:
                    opt = (index, i, score)
        return opt

    # Find the best combination
    opt = optimal(scores)
    best.append(({"metric": measures_str[opt[0]], "k": k_range[opt[1]]}, opt[2]))
    print(f"Best knn parameters: {best}")


def tree(train_X, train_y, test_X, test_y, **kwargs):
    # Define classifiers
    tree = DecisionTreeClassifier(**kwargs)

    # Fit the models
    tree.fit(train_X, train_y)

    # Compute the accuracy
    tree_pred = tree.predict(test_X)
    return accuracy_score(test_y, tree_pred)


@info
def tree_parameter_sweep(
    train_X,
    train_y,
    test_X,
    test_y,
    scores_depth,
    scores_split,
    scores_leaf,
    best,
    depth_range=range(1, 10),
    min_samples_split_range=range(2, 10),
    min_samples_leaf_range=range(1, 10),
):
    """
    Sweeps over the parameters max_depth, min_samples_split and min_samples_leaf
    """

    def listcomp(r, arg, return_value: List):
        print(f"Sweeping {arg} in {r}")
        ret = [
            score
            for _, score in itter(
                train_X,
                train_y,
                test_X,
                test_y,
                r,
                arg,
                target_func=tree,
            )
        ]
        return_value.extend(ret)

    start_and_wait(
        [
            Thread(
                target=listcomp,
                args=(depth_range, "max_depth", scores_depth),
            ),
            Thread(
                target=listcomp,
                args=(min_samples_split_range, "min_samples_split", scores_split),
            ),
            Thread(
                target=listcomp,
                args=(min_samples_leaf_range, "min_samples_leaf", scores_leaf),
            ),
        ]
    )

    # Find top 5 scores
    top_depth = top(scores_depth, depth_range)
    top_split = top(scores_split, min_samples_split_range)
    top_leaf = top(scores_leaf, min_samples_leaf_range)

    combined_scores = []
    for _, depth in top_depth:
        for _, split in top_split:
            for _, leaf in top_leaf:
                combined_scores.append(
                    (
                        {"depth": depth, "split": split, "leaf": leaf},
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
    best.append(sorted(combined_scores, key=lambda x: x[1], reverse=True)[0])
    print(f"Best tree parameters: {best}")


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
    forest = RandomForestClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
    )
    forest.fit(train_X, train_y)
    forest_pred = forest.predict(test_X)
    return accuracy_score(test_y, forest_pred)


@info
def random_forest_parameter_sweep(
    train_X,
    train_y,
    test_X,
    test_y,
    score_depth,
    score_split,
    score_est,
    best,
    depth_range=range(1, 10),
    min_samples_split_range=range(2, 10),
    n_estimators_range=range(1, 10),
):
    """
    Sweeps over the parameters max_depth, min_samples_split, and n_estimators
    """

    def list_comp(r, arg, return_value: List):
        ret = [
            score
            for _, score in itter(
                train_X,
                train_y,
                test_X,
                test_y,
                r,
                arg,
                target_func=random_forest,
            )
        ]
        return_value.extend(ret)

    start_and_wait([
        Thread(
            target=list_comp,
            args=(depth_range, "max_depth", score_depth),
        ),
        Thread(
            target=list_comp,
            args=(min_samples_split_range, "min_samples_split", score_split),
        ),
        Thread(
            target=list_comp,
            args=(n_estimators_range, "n_estimators", score_est),
        ),
    ])
    
    top_depth = top(depth_range, score_depth)
    top_split = top(min_samples_split_range, score_split)
    top_est = top(n_estimators_range, score_est)

    scores = []
    for depth, _ in top_depth:
        for split, _ in top_split:
            for est, _ in top_est:
                scores.append(
                    (
                        {"depth": depth, "split": split, "est": est},
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

    def optimal(scores):
        # We want to minimize the depth and estimators, but maximize the split and score

        eval = lambda x: x[1] / (x[0]["depth"] * 10 + x[0]["split"] + x[0]["est"] * 5)
        return sorted(scores, key=eval, reverse=True)[0]

    best.append(optimal(scores))
    print(f"Best random forest parameters: {best}")


if __name__ == "__main__":
    from task1 import suppress_qt_warnings
    from threading import Thread

    suppress_qt_warnings()
    train, test = load_data()
    k_range = range(5, 25)
    knn_ret, tree_ret, forest_ret = (
        ([[0 for _ in k_range] for _ in range(3)], []),
        ([], [], [], []),
        ([], [], [], []),
    )
    knn_parameters, tree_parameters, forest_parameters = (
        [range(5, 25)],
        [range(1, 30), range(2, 30), range(1, 30)],
        [range(1, 50), range(2, 50), range(1, 50)],
    )

    # Start the threads,
    threads = [
        Thread(
            target=knn_parameter_sweep,
            args=(
                *train,
                *test,
                *knn_ret,
                *knn_parameters,
            ),
        ),
        Thread(
            target=tree_parameter_sweep,
            args=(
                *train,
                *test,
                *tree_ret,
                *tree_parameters,
            ),
        ),
        Thread(
            target=random_forest_parameter_sweep,
            args=(
                *train,
                *test,
                *forest_ret,
                *forest_parameters,
            ),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Plot the scores
    # in 3 different figures

    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("KNN parameter sweep")
    plt.ylabel("accuracy")
    for i, score in enumerate(knn_ret[0]):
        plt.plot(k_range, score, label=["euclidean", "manhattan", "minkowski"][i])

    plt.legend()

    plt.figure()
    plt.title("Decision Tree parameter sweep")
    plt.ylabel("accuracy")
    plt.plot(range(1, 30), tree_ret[0], label="max_depth")
    plt.plot(range(2, 30), tree_ret[1], label="min_samples_split")
    plt.plot(range(1, 30), tree_ret[2], label="min_samples_leaf")
    plt.legend()

    plt.figure()
    plt.title("Random Forest parameter sweep")
    plt.ylabel("accuracy")
    plt.plot(range(1, 50), forest_ret[0], label="max_depth")
    plt.plot(range(2, 50), forest_ret[1], label="min_samples_split")
    plt.plot(range(1, 50), forest_ret[2], label="n_estimators")
    plt.legend()
    print(knn_ret[1])
    plt.show()
    with open("results.md", "w") as f:
        f.write(
            f"""
# Results

## KNN

### Best parameters

```python
metric  = {knn_ret[1][0][0]["metric"]}
k       = {knn_ret[1][0][0]["k"]}
```

### Best score

```python
score   = {knn_ret[1][0][1]}
```

## Decision Tree

### Best parameters

```python
max_depth           = {tree_ret[3][0][0]["depth"]}
min_samples_split   = {tree_ret[3][0][0]["split"]}
min_samples_leaf    = {tree_ret[3][0][0]["leaf"]}
```

### Best score

```python
score              = {tree_ret[3][0][1]}
```

## Random Forest

### Best parameters

The random forest best parameter depends
on some randomness, so the result may vary.
Our way of determining the best parameter
is $`score / (depth * 10 + split + est * 5)`$.
since we want to minimize the depth and estimators,
and don't really care about the split but want to maximize the score.


```python
max_depth           = {forest_ret[3][0][0]["depth"]}
min_samples_split   = {forest_ret[3][0][0]["split"]}
n_estimators        = {forest_ret[3][0][0]["est"]} 
```

### Best score

```python
score               = {forest_ret[3][0][1]}
```
"""
        )
    print("Done")
