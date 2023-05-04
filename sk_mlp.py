from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from task1 import preprocess

# define classifier
mlp = MLPClassifier()

# Load the data
to_drop, df = preprocess("./train-final.csv", corr_threshold=0.95)
test_df = preprocess("./test-final.csv", to_drop=to_drop)  # Drop the same columns

# Split the data into x and y
train_X = df.iloc[:, :-1].values.tolist()
train_y = df.iloc[:, -1].values.tolist()

# Split the data into x and y
test_X = test_df.iloc[:, :-1].values.tolist()
test_y = test_df.iloc[:, -1].values.tolist()

mlp.fit(train_X, train_y)

mlp_pred = mlp.predict(test_X)
print(f"MLP accuracy: {accuracy_score(test_y, mlp_pred)}")
