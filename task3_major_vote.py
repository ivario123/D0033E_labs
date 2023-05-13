from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from task1 import preprocess

if __name__ == "__main__":
    # Define classifiers
    # chosen the optimal parameters from task 2
    clf1 = DecisionTreeClassifier(max_depth=29,min_samples_split=2,min_samples_leaf=1)
    clf2 = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
    clf3 = MLPClassifier(hidden_layer_sizes=28, activation='tanh',solver='lbfgs',learning_rate='constant',random_state=1)
    clf4 = NuSVC(kernel='poly')
    clf5 = RandomForestClassifier(max_depth=28,min_samples_split=5,n_estimators=47)
    ensemble = VotingClassifier(
    estimators=[('decisiontree', clf1), ('knn', clf2), ('mlp', clf3), ('svm', clf4), ('randomforest', clf5)])

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
    ensemble.fit(train_X, train_y)

    # predicting the output on the test dataset
    pred_final = ensemble.predict(test_X)
    print(f" Accuracy: {accuracy_score(test_y, pred_final)}")
