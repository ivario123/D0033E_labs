import numpy as np
from typing import List, Tuple
from scipy.stats import entropy


class TreeNode:
    def __init__(
        self,
        cutoff: float,
        index: int = 0,
        left: "TreeNode" = None,
        right: "TreeNode" = None,
    ) -> None:
        self.left = left
        self.index = index
        self.right = right
        self.cutoff = cutoff

    def traverse(self, x: List) -> any:
        if type(x) == float:
            print(f"Traversing with {x}")
            assert False
        if self.left is None and self.right is None:
            return self.cutoff
        if x[self.index] < self.cutoff:
            return self.left.traverse(x)
        else:
            return self.right.traverse(x)

    def __str__(self) -> str:
        return f"TreeNode({self.left}, {self.right}, {self.cutoff})"


def normcorr(x: List, y: List) -> float:
    """
    Computes the normalized correlation between two vectors
    """
    x = np.array(x)
    y = np.array(y)
    return np.corrcoef(x, y)[0, 1]


class TreeClassifier:
    def __init__(self, max_d=5, min_samples_split=2) -> None:
        self.d = max_d
        self.root = None
        self.min_samples_split = min_samples_split

    def fit(self, X: List[List], y: List) -> None:
        """
        Fits the tree to the given data
        """
        self.root = self.build_tree(X, y, 0)

    def build_tree(self, X: List[List], y: List, depth: int) -> TreeNode:
        """
        Builds the tree recursively
        """
        index, split_value = self.best_split(X, y)
        if len(y) <= self.min_samples_split:
            return TreeNode(np.max(y))
        if depth >= self.d:
            return TreeNode(np.max(y))
        else:
            dl = [X[i] for i in range(0, len(X)) if X[i][index] < split_value]
            dr = [X[i] for i in range(0, len(X)) if X[i][index] >= split_value]
            y_l = [y[i] for i in range(0, len(X)) if X[i][index] < split_value]
            y_r = [y[i] for i in range(0, len(X)) if X[i][index] >= split_value]
            if len(y_l) < self.min_samples_split:
                if len(y_l) == 0:
                    return -1
                mean = np.max(y_l)
                left_node = TreeNode(mean, index=index)
            else:
                left_node = self.build_tree(dl, y_l, depth + 1)
                if left_node == -1:
                    left_node = TreeNode(np.max(y_l))
            if len(y_r) < self.min_samples_split:
                if len(y_r) == 0:
                    return -1
                mean = np.max(y_r)
                right_node = TreeNode(mean, index=index)
            else:
                right_node = self.build_tree(dr, y_r, depth + 1)
                if right_node == -1:
                    right_node = TreeNode(np.max(y_r))
        return TreeNode(split_value, index=index, left=left_node, right=right_node)

    def entropy(self, y: List) -> float:
        """
        Computes the entropy of the given data
        """
        # use scipy to compute the entropy
        _, counts = np.unique(y, return_counts=True)
        return entropy(counts, base=None)

    def info_gain(self, y: List, left: List, right: List) -> float:
        """
        Computes the information gain of the given data
        """
        entropy = self.entropy(y)
        ent_l, w_l = self.entropy(left), len(left) / len(y)
        ent_r, w_r = self.entropy(right), len(right) / len(y)
        return entropy - (w_l * ent_l + w_r * ent_r)

    def best_split(self, X: List[List], y: List) -> Tuple[int, float]:
        """
        Finds the best split for the given data
        """

        optimal_split = (0, 0, float("-inf"))
        if len(X) == 0:
            return optimal_split[:-1]
        for index in range(0, len(X[0])):
            # Loop over all possible splits
            unique_values = np.unique(np.array(X)[:, index])

            for potential_split in range(0, len(unique_values)):
                # Split the data
                left_y = [
                    y[i]
                    for i in range(0, len(X))
                    if X[i][index] < unique_values[potential_split]
                ]
                right_y = [
                    y[i]
                    for i in range(0, len(X))
                    if X[i][index] >= unique_values[potential_split]
                ]
                info_gain = self.info_gain(y, left_y, right_y)
                if info_gain > optimal_split[2]:
                    optimal_split = (index, unique_values[potential_split], info_gain)
        return optimal_split[:-1]

    def estimate(self, x: List) -> any:
        """
        Estimates the value of the given data
        """
        if self.root is None:
            raise Exception("Tree not fit")
        if type(x) is not list:
            raise Exception("Input must be a list")
        return self.root.traverse(x)


def print_tree(tree: TreeNode, depth: int = 0) -> None:
    def traverse(tree: TreeNode, depth: int = 0) -> List[Tuple[TreeNode, int]]:
        ret = [(tree, depth)]
        if tree.left is not None:
            ret += traverse(tree.left, depth + 1)
        if tree.right is not None:
            ret += traverse(tree.right, depth + 1)
        return ret

    tree_map = {}
    for node, depth in traverse(tree):
        if depth not in tree_map:
            tree_map[depth] = []
        tree_map[depth].append(node)
    # Center
    lines = []
    for depth in tree_map:
        # Line start
        repr = ""
        for i, node in enumerate(tree_map[depth]):
            if i != 0:
                repr += " " * (len(tree_map[depth]) - 1) * 2
            repr += f"{node.cutoff}"
        lines.append(repr)
    # Indent to center based on longest line
    longest_line = max(lines, key=lambda x: len(x))
    for i, line in enumerate(lines):
        # Add spaces to the front of the line
        lines[i] = " " * (len(longest_line) - len(line) // 2) + line
    return "\n".join(lines)


from random import randint


def accuracy(tree: TreeClassifier, X: List[List], y: List) -> float:
    """
    Computes the accuracy of the given tree
    """
    correct = 0
    for i in range(0, len(X)):
        est = tree.estimate(X[i])
        print(f"Estimate: {est}, Actual: {y[i]}")
        if est == y[i]:
            correct += 1

    return correct / len(X)


from pandas import read_csv

if __name__ == "__main__":
    # data = read_csv("data_processed.csv")
    # Read data from the csv using pandas
    df = read_csv("data_processed.csv")

    # We don't need the text version of the label
    #df = df.drop("gesture label", axis=1)

    # Replace missing values with mean of that column
    df = df.fillna(df.mean())
    data = df.values.tolist()
    labels = [row.pop() for row in data]  # Remove the label from the data
    data = data[:100]  # Only use the first 100 rows due to performance issues

    data = [[float(x) for x in row] for row in data]  # Convert to floats
    tree = TreeClassifier(max_d=5)
    tree.fit(data, labels)
    with open("tree.txt", "w") as f:
        f.write(print_tree(tree.root))

    print(accuracy(tree, data, labels))
