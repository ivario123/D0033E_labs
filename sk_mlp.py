from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import csv

from task1 import preprocess


f=open('mlp.csv','w')
writer=csv.writer(f)

# define classifier
mlp = MLPClassifier()

# Load the data
to_drop, df = preprocess("./train-final.csv", corr_threshold=0.95)
test_df = preprocess("./test-final.csv", to_drop=to_drop)  # Drop the same columns

# Split the data into x and y
train_x = df.iloc[:, :-1].values.tolist()
train_y = df.iloc[:, -1].values.tolist()

# Split the data into x and y
test_x = test_df.iloc[:, :-1].values.tolist()
test_y = test_df.iloc[:, -1].values.tolist()


parameter_space = {
    'hidden_layer_sizes': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100],
    'activation': ['tanh', 'relu', 'logistic', 'identity'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    'random_state': [1, None],
}

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(train_x, train_y)

# Best paramete set
print('Best parameters:\n', clf.best_params_)
writer.writerow(['Best parameters:\n', clf.best_params_])

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    str="%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params)
    writer.writerow([str])


f.close()