#CS 412 Homework 4 Submission Stub
#Zakir Ujjainwala

from os import getpgid
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_splits(n, k):
    # Making a sublist out of the list
    sub = []
    result = []
    all_indices = [item for item in range(n)]
    np.random.shuffle(all_indices)
    length = (-1 * len(all_indices) // k * -1)

    for i in all_indices:
        sub.append(i)
        if len(sub) == length:
            result.append(sub)
            sub = []
    if sub:
        result.append(sub)
    print(result)

    return result

    # return [[0,2], [1,3]]

def my_cross_val(method, X, y, k):

    if method == 'LinearSVC':
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score
        # Create the model
        myLinSVC = LinearSVC(max_iter=5000).fit(X, y)

        # have to recreate this function, cannot use cross_val_score
        # Perform the k-fold cross validation
        accuracyLinSVC = cross_val_score(myLinSVC,X,y,cv=10)
        print(accuracyLinSVC)
        print(np.mean(accuracyLinSVC))

        return 0

    elif method == 'SVC':
        return 0

    elif method == 'LogisticRegression':
        return 0

    elif method == 'RandomForestClassifier':
        return 0

    elif method == 'XGBClassifier':
        return 0

    return np.array([1]*k)

def my_train_test(method, X, y, pi, k):

    return np.array([1]*k)

# get_splits(11,3)
# get_splits(5,2)
# get_splits(4,2)
# get_splits(5,3)