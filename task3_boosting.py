from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


from task1 import preprocess

if __name__ == "__main__":
    # Define classifiers
    # maybe change parameters?
    boosting = GradientBoostingClassifier(max_depth = None)

    # Load the data
    to_drop, df = preprocess("./train-final.csv", corr_threshold=0.95)
    test_df = preprocess("./test-final.csv", to_drop=to_drop)  # Drop the same columns

    # Split the data into x and y
    train_X = df.iloc[:, :-1].values.tolist()
    train_y = df.iloc[:, -1].values.tolist()

    # Split the data into x and y
    test_X = test_df.iloc[:, :-1].values.tolist()
    test_y = test_df.iloc[:, -1].values.tolist()

    # training all the model on the train dataset
    boosting.fit(train_X, train_y)

    # predicting the output on the test dataset
    pred_final = boosting.predict(test_X)
    print(f" Accuracy: {accuracy_score(test_y, pred_final)}")

# acc 0.5685185185185185