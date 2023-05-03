# Uses sklearn to implement KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

from task1 import preprocess

if __name__ == "__main__":
    # Define classifiers
    knn = KNeighborsClassifier(n_neighbors=5)

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
    knn.fit(train_X, train_y)

    # Compute the accuracy
    knn_pred = knn.predict(test_X)
    print(f"KNN accuracy: {accuracy_score(test_y, knn_pred)}")
