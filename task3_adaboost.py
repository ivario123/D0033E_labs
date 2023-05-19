from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from task1 import preprocess


# Define classifiers
clf1 = DecisionTreeClassifier(max_depth=29,min_samples_split=2,min_samples_leaf=1)
clf5 = RandomForestClassifier(max_depth=28,min_samples_split=5,n_estimators=50)
# maybe change parameters?
boosting = AdaBoostClassifier(clf5)
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
