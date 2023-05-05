from threading import Thread
from typing import Any, Callable, List, Tuple

from pandas import DataFrame as df
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from knn import EUCLIDEAN
from sk_tree import DecisionTreeClassifier
from task1 import preprocess
import time
from sklearn import svm

NUM_SAMPLES = 1


def load_data(
    test="./test-final.csv", train="./train-final.csv", corr_threshold=0.75, **kwargs
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
    """
    returns ((train_X, train_y), (test_X, test_y))
    """
    # Load the data
    to_drop, df = preprocess(train, corr_threshold=corr_threshold, **kwargs)
    # Drop the same columns
    test_df = preprocess(test, to_drop=to_drop, **kwargs)  # Drop the same columns

    # Check if the data is not the same
    def eq(x, y):
        for col in x.columns:
            if col != "class":
                yield (x[col].values == y[col].values).all()

    assert not all(eq(df, test_df)), "Train and test data are the same"

    # Split the data into x and y
    train = df.iloc[:, :-1].values.tolist(), df.iloc[:, -1].values.tolist()

    # Split the data into x and y
    test = test_df.iloc[:, :-1].values.tolist(), test_df.iloc[:, -1].values.tolist()
    return train, test


def info(func: Callable):
    offset = lambda x, y: y // 2 - x // 2 if y > x else 0
    l = 50
    name = func.__name__.replace("_", " ").upper()

    def wrapper(*args, **kwargs):
        print("=" * l)

        print(f"{' '*offset(len(name),l)}{name}")
        print("=" * l)
        start_time = time.time()
        # Functions decorated with @info should not return anything
        func(*args, **kwargs)
        end_time = time.time()
        print("-" * l)
        end_msg = f"function {name} finished in {(end_time - start_time):.1f} seconds"
        print(f"{' '*(offset(len(str(end_msg)),l))}{end_msg}")
        print("-" * l)

    return wrapper


from statistics import mean


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
        if target_func == tree:
            scores = [
                target_func(train_X, train_y, test_X, test_y, **{arg: val})
                for _ in range(NUM_SAMPLES)
            ]

            def strip(x):
                return [i[0] for i in x], [i[1] for i in x]

            scores, depths = strip(scores)

            score = mean([float(score) for score in scores])
            depths = int(mean(depths))
            yield i, (score, depths)
        else:
            score = mean(
                [
                    float(target_func(train_X, train_y, test_X, test_y, **{arg: val}))
                    for _ in range(NUM_SAMPLES)
                ]
            )
            yield i, score


top = lambda x, y: sorted(
    zip(x, y), key=lambda x: x[1][0] if type(x[1]) == list else x[1], reverse=True
)[:3]


def start_and_wait(threads: List[Thread]):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def svm_func(X, y, X_t, y_t):
    clf = svm.LinearSVC(C=1.0, multi_class="ovr")
    clf.fit(X, y)  # training data
    prediction = clf.predict(X_t)
    acc = accuracy_score(y_t, prediction)
    return acc


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
    measures_str = ["euclidean", "manhattan"]

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
    return accuracy_score(test_y, tree_pred), tree.get_depth()


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
    combined_scores.append(
        (
            {"depth": top_depth[0][1], "split": 2, "leaf": 1},
            tree(
                train_X,
                train_y,
                test_X,
                test_y,
                max_depth=top_depth[0][1],
            ),
        )
    )
    combined_scores.append(
        (
            {"depth": 5, "split": top_split[0][1], "leaf": 1},
            tree(
                train_X,
                train_y,
                test_X,
                test_y,
                min_samples_split=top_split[0][1],
            ),
        )
    )
    combined_scores.append(
        (
            {"depth": 5, "split": 2, "leaf": top_leaf[0][1]},
            tree(
                train_X,
                train_y,
                test_X,
                test_y,
                min_samples_leaf=top_leaf[0][1],
            ),
        )
    )
    score = lambda x: x[1][0]

    def optimal(scores):
        # We want to minimize the actual depth but keep the score high
        ordered = sorted(scores, key=score, reverse=True)
        return sorted(
            [i for i in ordered if i[1][0] == ordered[0][1][0]], key=lambda x: x[1][1]
        )[0]

    best.append(sorted(combined_scores, key=score, reverse=True)[0])
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

    start_and_wait(
        [
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
        ]
    )

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
        eval = lambda x: x[
            1
        ]  # / (x[0]["depth"] * 10 + x[0]["split"] + x[0]["est"] * 5)
        ordered = sorted(scores, key=eval, reverse=True)
        return sorted(
            [i for i in ordered if i[1] == ordered[0][1]], key=lambda x: x[0]["depth"]
        )[0]

    best.append(optimal(scores))
    print(f"Best random forest parameters: {best}")


def corr_test(corr_range: range = range(50, 95, 2)):
    # Plot the results
    def fig(name):
        plt.figure(name)
        plt.title(name)
        plt.ylabel("accuracy")

    def end_fig(name, corr, drop_below_spine=""):
        plt.legend()
        plt.savefig(
            image_folder
            + f"/{name}_{NUM_SAMPLES}_{int(corr*100)}%_{drop_below_spine}.png"
        )

    OKGREEN = "\033[92m"
    ENDC = "\033[0m"
    best_results = {
        "knn": {"acc": float("-inf"), "params": [], "corr": 0, "legs": False},
        "tree": {"acc": float("-inf"), "params": [], "corr": 0, "legs": False},
        "forest": {"acc": float("-inf"), "params": [], "corr": 0, "legs": False},
    }
    corr_effect_with_legs = [[], [], []]
    corr_effect_without_legs = [[], [], []]
    for drop_below_spine in [True,False]:
        for corr in corr_range:
            print("\n" * 5)
            print(f"{OKGREEN}Correlation threshold: {corr}%{ENDC}")
            print("\n" * 5)
            corr = corr / 100
            # Retrieve the data from the csv files and preprocess them
            train, test = load_data(
                corr_threshold=corr, drop_below_spine=drop_below_spine
            )
            acc = svm_func(*train, *test)
            print(f"Accuracy SVM: {acc:.5f}")

            # exit()
            # Define input and output parameters
            k_range = range(5, 25)
            knn_ret, tree_ret, forest_ret = (
                ([[0 for _ in k_range] for _ in range(2)], []),
                ([], [], [], []),
                ([], [], [], []),
            )
            knn_parameters, tree_parameters, forest_parameters = (
                [k_range],  # k
                [range(1, 30), range(2, 30), range(1, 30)],  # depth, split, leaf
                [range(1, 50), range(2, 50), range(1, 50)],  # depth, split, est
            )
            t = lambda f, ret, param: Thread(
                target=f, args=(*train, *test, *ret, *param)
            )
            # Start the threads, one for each algorithm
            start_and_wait(
                [
                    #  t(svm_func,svm_ret,svm_parameters),
                    t(knn_parameter_sweep, knn_ret, knn_parameters),
                    t(tree_parameter_sweep, tree_ret, tree_parameters),
                    t(random_forest_parameter_sweep, forest_ret, forest_parameters),
                ]
            )
            image_folder = "./images"

            fig("KNN parameter sweep")
            plt.xlabel("k")
            for i, score in enumerate(knn_ret[0]):
                plt.plot(k_range, score, label=["euclidean", "manhattan"][i])
            end_fig("knn", corr, drop_below_spine)

            fig("Decision Tree parameter sweep")
            strip = lambda x: [i[0] for i in x]
            for i in range(len(tree_parameters)):
                plt.plot(
                    tree_parameters[i],
                    strip(tree_ret[i]),
                    label=["depth", "split", "leaf"][i],
                )
            end_fig("tree", corr, drop_below_spine)

            fig("Random Forest parameter sweep")
            for i in range(len(forest_parameters)):
                plt.plot(
                    forest_parameters[i],
                    forest_ret[i],
                    label=["depth", "split", "est"][i],
                )
            end_fig("forest", corr, drop_below_spine)
            plt.close("all")
            # Write the results to a file
            env = Environment(loader=FileSystemLoader("."))
            template = env.get_template("result_template.md.jinja")
            template.stream(
                tree_ret=tree_ret,
                knn_ret=knn_ret,
                forest_ret=forest_ret,
            ).dump(f"results_{corr}_{drop_below_spine}.md")

            # Save the best results
            # KNN
            knn_best = knn_ret[-1][0]
            if knn_best[1] > best_results["knn"]["acc"]:
                best_results["knn"] = {
                    "acc": knn_best[1],
                    "corr": corr,
                    "params": knn_best[0],
                    "legs": drop_below_spine,
                }
            # Decision Tree
            tree_best = tree_ret[-1][0]
            if tree_best[1][0] > best_results["tree"]["acc"]:
                best_results["tree"] = {
                    "acc": tree_best[1][0],
                    "corr": corr,
                    "params": tree_best[0],
                    "legs": drop_below_spine,
                }
            # Random Forest
            forest_best = forest_ret[-1][0]
            if forest_best[1] > best_results["forest"]["acc"]:
                best_results["forest"] = {
                    "acc": forest_best[1],
                    "corr": corr,
                    "params": forest_best[0],
                    "legs": drop_below_spine,
                }
            for i, (c, score) in enumerate(
                zip(
                    corr_effect_with_legs
                    if drop_below_spine
                    else corr_effect_without_legs,
                    [knn_best, tree_best, forest_best],
                )
            ):
                c.append(score[1] if i != 1 else score[1][0])

            print("Done")
    import json

    with open("best_results.json", "w") as f:
        json.dump(best_results, f)
    fig("Correlation effect")
    plt.xlabel("Correlation threshold")
    plt.plot(corr_range, corr_effect_with_legs[0], label="KNN_legs")
    plt.plot(corr_range, corr_effect_with_legs[1], label="Decision Tree_legs")
    plt.plot(corr_range, corr_effect_with_legs[2], label="Random Forest_legs")
    plt.plot(corr_range, corr_effect_without_legs[0], label="KNN_no legs")
    plt.plot(corr_range, corr_effect_without_legs[1], label="Decision Tree_no legs")
    plt.plot(corr_range, corr_effect_without_legs[2], label="Random Forest_no legs")
    end_fig("corr_effect", 0)
    plt.show()


if __name__ == "__main__":
    from threading import Thread

    import matplotlib.pyplot as plt
    from jinja2 import Environment, FileSystemLoader

    from task1 import suppress_qt_warnings

    # qt5 warnings clutter up the output
    suppress_qt_warnings()
    corr_test()
