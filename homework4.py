#CS 412 Homework 4 Submission Stub
#Zakir Ujjainwala

import numpy as np
import random

# Test database
from sklearn.datasets import load_digits
digits = load_digits()
# X, y = digits.data, digits.target

def get_splits(n, k):
    # Making a sublist out of the list - First Attempt
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

    # Second Attempt
    # dataSplit = list()
    # dataset = [item for item in range(n)]
    # #print(dataset)
    # dataCopy = list(dataset)
    # #print(dataCopy)
    # foldSize = int(len(dataCopy) / k)
    # for i in range(k):
    #     fold = list()
    #     while len(fold) < foldSize:
    #         index = random.randrange(len(dataCopy))
    #         fold.append(dataCopy.pop(index))
    #     dataSplit.append(fold)
    # #print(dataSplit)
    # return dataSplit
    # return [[0,2], [1,3]]
    

# get_splits(11,3)
# get_splits(7,2)
# get_splits(4,2)
# get_splits(5,3)
# get_splits(10,4)
# get_splits(15,10)

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
        return np.array([1]*k)

# my_cross_val('LinearSVC', [0], 3, k=15)

def my_train_test(method, X, y, pi, k):
    X, y = digits.data, digits.target
    # n_train = int(pi * n_samples)
    # n_test = int(n_samples - n_train)
    scores = list()
    for i in range(k):

        Xn_samples = len(X)
        X_train = list()
        Xn_train = pi * Xn_samples
        X_test = list(X)
        while len(X_train) < Xn_train:
            index = random.randrange(len(X_test))
            X_train.append(X_test.pop(index))
        # print(X_train, X_test)

        yn_samples = len(y)
        y_train = list()
        yn_train = pi * yn_samples
        y_test = list(y)
        while len(y_train) < yn_train:
            ind = random.randrange(len(y_test))
            y_train.append(y_test.pop(index))

        if method == 'LinearSVC':
            from sklearn.svm import LinearSVC
            # Create the model
            myLinSVC = LinearSVC(max_iter=5000).fit(X_train, y_train)
            scores.append(myLinSVC.score(X_test, y_test))
        
        elif method == 'SVC':
            from sklearn.svm import SVC
            # Create the model
            mySVC = SVC(gamma='scale', C=10).fit(X_train, y_train)

        elif method == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            # Create the model
            myLGR = LogisticRegression(penalty='l2', solver='lbfgs',
            multi_class='multinomial').fit(X_train, y_train)

        elif method == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            # Create the model
            myRFC = RandomForestClassifier(max_depth=20, random_state=0,
            n_estimators=500).fit(X_train, y_train)

        elif method == 'XGBClassifier':
            from xgboost import XGBClassifier
            # Create the model
            myXGB = XGBClassifier(max_depth=5).fit(X_train, y_train)

        else:
            return np.array([1]*k)
        
        return scores

# my_train_test('LinearSVC', [0], [1], 0.75, 10)