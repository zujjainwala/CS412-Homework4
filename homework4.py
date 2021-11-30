#CS 412 Homework 4 Submission Stub
#Zakir Ujjainwala

import numpy as np
import sklearn
import random
from itertools import permutations

# Test database
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target

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
    # print(result)

    return result

    # return [[0,2], [1,3]]

def my_cross_val(method, X, y, k):

    split_data = get_splits(len(X), k)
    
    results = []

    if method == 'LinearSVC':
        from sklearn.svm import LinearSVC
        # Create the model
        myLinSVC = LinearSVC(max_iter=5000)

        # Perform the k-fold cross validation
        # accuracyLinSVC = cross_val_score(myLinSVC,X,y,cv=10)
        
        #print(accuracyLinSVC)
        #print(np.mean(accuracyLinSVC))

        return 0

    elif method == 'SVC':
        from sklearn.svm import SVC

        # Create the model
        mySVC = SVC(gamma='scale', C=10)

        # Perform the k-fold cross validation
        return 0

    elif method == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression

        # Create the model
        myLGR = LogisticRegression(penalty='l2', solver='lbfgs',
        multi_class='multinomial')

        # Perform the k-fold cross validation
        return 0

    elif method == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier

        # Create the model
        myRFC = RandomForestClassifier(max_depth=20, random_state=0,
        n_estimators=500)

        # Perform the k-fold cross validation
        return 0

    elif method == 'XGBClassifier':
        from xgboost import XGBClassifier

        # Create the model
        myXGB = XGBClassifier(max_depth=5)

        # Perform the k-fold cross validation
        return 0

    else:
        # Default
        return np.array([1]*k)

def my_train_test(method, X, y, pi, k):
    X, y = digits.data, digits.target
    n_samples = len(X)
    train_size = pi*n_samples
    test_size = 1 - train_size

    for i in range(k):
        return i

    if method == 'LinearSVC':
        return 0
    
    elif method == 'SVC':
        return 0

    elif method == 'LogisticRegression':
        return 0

    elif method == 'RandomForestClassifier':
        return 0

    elif method == 'XGBClassifier':
        return 0

    else:
        return np.array([1]*k)

# get_splits(11,3)
# get_splits(5,2)
# get_splits(4,2)
# get_splits(5,3)
# my_cross_val('LinearSVC', [0], 3, k=15)
my_train_test('LinearSVC', [0], [1], 0.75, 10)