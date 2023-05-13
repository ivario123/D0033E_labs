from threading import Thread
from typing import Any, Callable, List, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC
from sk_tree import DecisionTreeClassifier
from task1 import preprocess
import time

NUM_SAMPLES = 5
IMAGE_FOLDER = "./images"
OKGREEN = "\033[92m"
ENDC = "\033[0m"


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


def start_and_wait(*threads: Thread):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


thread = lambda *args: start_and_wait(
    *[Thread(target=target, args=args) for target, args in args]
)


def svm_func(X, y, X_t, y_t, **kwargs):
    svm = NuSVC(**kwargs)  # kernel='rbf', gamma='scale')
    svm.fit(X, y)

    # clf = svm.NuSVC(C=c, multi_class="ovr",max_iter=5000)
    # clf.fit(X, y)  # training data
    prediction = svm.predict(X_t)
    acc = accuracy_score(y_t, prediction)
    return acc


@info
def svm_parameter_sweep(
    train_X,
    train_y,
    test_X,
    test_y,
    scores,
    randome,
    best,
    gamma=range(3, 15),
    kernels=["linear", "poly", "rbf"],
    # c_range=range(5, 20),
):
    print(kernels)

    def compute(*args, **kwargs):
        ret = svm_func(*args, **kwargs)
        return ret

    tmp_scores = {}
    for k in kernels:
        tmp_scores[k] = [
            compute(train_X, train_y, test_X, test_y, gamma=g, kernel=k) for g in gamma
        ]
    scores.append(tmp_scores)
    top = ({"kernel": "", "gamma": 0}, float("-inf"))
    for k in kernels:
        if max(tmp_scores[k]) > top[1]:
            top = (
                {"k": k, "gamma": gamma[tmp_scores[k].index(max(tmp_scores[k]))]},
                max(tmp_scores[k]),
            )
    best.append(top)
    print(f"{top=}")

def knn(train_X, train_y, test_X, test_y, n_neighbors=5, distance_measure=None):
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
        *[
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

    thread(
        (listcomp, (depth_range, "max_depth", scores_depth)),
        (listcomp, (min_samples_split_range, "min_samples_split", scores_split)),
        (listcomp, (min_samples_leaf_range, "min_samples_leaf", scores_leaf)),
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

    thread(
        (list_comp, (depth_range, "max_depth", score_depth)),
        (list_comp, (min_samples_split_range, "min_samples_split", score_split)),
        (list_comp, (n_estimators_range, "n_estimators", score_est)),
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


@info
def corr_test(corr_range: range = range(94, 100, 2)):
    # Helper functions

    def fig(name):
        """
        Makes a new figure and gives it a title and y label
        """
        plt.figure(name)
        plt.title(name)
        plt.ylabel("accuracy")

    def end_fig(name, corr, drop_below_spine=""):
        """
        Adds a legend and saves the figure
        """
        plt.legend()
        plt.savefig(
            IMAGE_FOLDER
            + f"/{name}_{NUM_SAMPLES}_{int(corr*100)}%_{drop_below_spine}.png"
        )

    t = lambda f, ret, param: Thread(target=f, args=(*train, *test, *ret, *param))

    # * Add (function_name, string_repr) here for each function you want to sweep over
    funcs, func_names = list(
        zip(
            *[
                (knn_parameter_sweep, "knn"),
                (tree_parameter_sweep, "tree"),
                (random_forest_parameter_sweep, "forest"),
                (svm_parameter_sweep, "svm"),
            ]
        )
    )

    # Global outputs
    best_results = {}
    #
    corr_effect_with_legs, corr_effect_without_legs = (
        [[] for _ in range(len(func_names))]
        for _ in range(2)  # One for each function and one for each drop_below_spine
    )

    # Parameter definitions
    k_range = range(5, 25)
    # * Add parameters here, one for each that you sweep over
    params = (
        [k_range],  # k
        [range(1, 30), range(2, 30), range(1, 30)],  # depth, split, leaf
        [range(1, 50), range(2, 50), range(1, 50)],  # depth, split, est
        [["linear", "poly", "rbf"], range(3, 15)],
    )
    # * Add name of parameters here, one for each that you sweep over
    param_names = (
        ["k"],
        ["depth", "min_samples_split", "min_samples_leaf"],
        ["depth", "min_samples_split", "n_estimators"],
        ["kernel", "gamma"],
    )
    _, tree_parameters, forest_parameters, svm_parameters = params

    for drop_below_spine in [True, False]:
        for corr in corr_range:
            print("\n" * 5)
            print(f"{OKGREEN}Correlation threshold: {corr}%{ENDC}")
            print("\n" * 5)
            corr = corr / 100
            # Retrieve the data from the csv files and preprocess them
            train, test = load_data(
                corr_threshold=corr, drop_below_spine=drop_below_spine
            )
            #    acc = svm_func(*train, *test)
            #    print(f"Accuracy SVM: {acc:.5f}")

            # define result arrays
            # * Add output arrays here, one for each output and one for the best combination
            knn_ret, tree_ret, forest_ret, svm_ret = (
                ([[0 for _ in k_range] for _ in range(2)], []),
                ([], [], [], []),
                ([], [], [], []),
                ([], []),
            )
            # * Add functions to thread list
            # Since pythons alias system is strange we can't do this in a generic way
            threads = [
                t(funcs[0], knn_ret, params[0]),
                t(funcs[1], tree_ret, params[1]),
                t(funcs[2], forest_ret, params[2]),
                t(funcs[3], svm_ret, params[3]),
            ]

            # Start the threads, one for each algorithm
            start_and_wait(*threads)
            # * Add results here, one for each function
            # This can't be done generically because of the aliasing system
            rets = [knn_ret, tree_ret, forest_ret, svm_ret]

            # Plot lambda function to clean up the code
            plot = lambda params, acc, label, r: [
                plt.plot(params[i], acc[i], label=label[i]) for i in r
            ]

            # * Add plots here, one for each function
            fig("KNN parameter sweep")
            plt.xlabel("k")
            for i, score in enumerate(knn_ret[0]):
                plt.plot(k_range, score, label=["euclidean", "manhattan"][i])
            end_fig("knn", corr, drop_below_spine)

            fig("Decision Tree parameter sweep")
            strip = lambda x: [[i[0] for i in x] for x in x]
            plot(
                tree_parameters,
                strip(tree_ret),
                param_names[1],
                range(len(tree_parameters)),
            )
            end_fig(func_names[1], corr, drop_below_spine)

            fig("Random Forest parameter sweep")
            plot(
                forest_parameters,
                forest_ret,
                param_names[2],
                range(len(forest_parameters)),
            )
            end_fig(func_names[2], corr, drop_below_spine)
            # plt.close("all")
            tmp = svm_ret[0][0]
            lin, pol, rbf = [
                tmp["linear"],
                tmp["poly"],
                tmp["rbf"],
            ]
            fig("SVM sweep")

            plt.plot(svm_parameters[-1], lin, label="linear")
            plt.plot(svm_parameters[-1], pol, label="poly")
            plt.plot(svm_parameters[-1], rbf, label="rbf")

            end_fig(func_names[3], corr, drop_below_spine)

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
            continue
            overall_best = [
                x[-1][0] for x in rets
            ]  # Listcomprehension not strictly necessary, but it makes the code more readable imo
            score_lookup = [
                lambda x: x[1],
                lambda x: x[1][0],
                lambda x: x[1],
                lambda x: x[1],
            ]
            best = [
                (label, score_lookup[i](score))
                for i, (label, score) in enumerate(zip(func_names, overall_best))
            ]
            # Store the best results
            for i, (label, score) in enumerate(best):
                if label not in best_results.keys():
                    best_results[label] = {
                        "acc": score,
                        "corr": corr,
                        "params": overall_best[i][0],
                        "legs": drop_below_spine,
                    }
                elif score > best_results[label]["acc"]:
                    best_results[label] = {
                        "acc": score,
                        "corr": corr,
                        "params": overall_best[i][0],
                        "legs": drop_below_spine,
                    }

            # Store in the appropriate arrays
            for i, (c, score) in enumerate(
                zip(
                    corr_effect_with_legs
                    if drop_below_spine
                    else corr_effect_without_legs,
                    overall_best,
                )
            ):
                c.append(score[1] if i != 1 else score[1][0])

            print("Done")
    return
    import json

    # Dump the best results to a file
    with open("best_results.json", "w") as f:
        json.dump(best_results, f)

    # Plot the results for the correlation effect
    fig("Correlation effect")
    plt.xlabel("Correlation threshold")
    plot_similar = lambda accs, labels: [
        plt.plot(corr_range, accs[i], label=labels[i]) for i in range(len(accs))
    ]
    plot_similar(
        corr_effect_with_legs,
        [x.upper() + " with legs" for x in func_names],
    )
    plot_similar(
        corr_effect_without_legs,
        [x.upper() + " without legs" for x in func_names],
    )
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
