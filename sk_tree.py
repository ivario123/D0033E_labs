# Tree classifier using sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

from task1 import preprocess

if __name__ == "__main__":
    # Define classifiers
    tree = DecisionTreeClassifier()
    forest = RandomForestClassifier()

    # Load the data
    to_drop, df = preprocess("./train-final.csv", corr_threshold=0.95)
    test_df = preprocess("./test-final.csv", to_drop=to_drop)  # Drop the same columns

    # Split the data into x and y
    train_X = df.iloc[:, :-1].values.tolist()
    train_y = df.iloc[:, -1].values.tolist()

    # Split the data into x and y
    test_X = test_df.iloc[:, :-1].values.tolist()
    test_y = test_df.iloc[:, -1].values.tolist()

    # Fit the models
    tree.fit(train_X, train_y)
    forest.fit(train_X, train_y)

    # Compute the accuracy
    tree_pred = tree.predict(test_X)
    forest_pred = forest.predict(test_X)
    print(f"Tree accuracy: {accuracy_score(test_y, tree_pred)}")
    print(f"Forest accuracy: {accuracy_score(test_y, forest_pred)}")
