#CS 412 Homework 4 Submission Stub
#Zakir Ujjainwala

import numpy as np
import random
from sklearn.metrics import accuracy_score

# Test database
from sklearn.datasets import load_digits
digits = load_digits()

def get_splits(n, k):
    all_indices = [item for item in range(n)]
    np.random.shuffle(all_indices)
    n_samples = len(all_indices)
    length1 = n_samples // k + 1
    length2 = n_samples // k

    sub = []
    result = []
    count = 1
    for i in all_indices:
        if count <= (n_samples % k):
            sub.append(i)
            if len(sub) == length1:
                result.append(sub)
                count += 1
                sub = []
        else:
            sub.append(i)
            if len(sub) == length2:
                result.append(sub)
                count += 1
                sub = []
    # if sub:
    #     result.append(sub)
    return result
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

    # if method == 'LinearSVC':
    #     from sklearn.svm import LinearSVC
    #     # Create the model
    #     myLinSVC = LinearSVC(max_iter=5000)

    #     # Perform the k-fold cross validation
    #     # accuracyLinSVC = cross_val_score(myLinSVC,X,y,cv=10)
        
    #     #print(accuracyLinSVC)
    #     #print(np.mean(accuracyLinSVC))

    #     return 0

    # elif method == 'SVC':
    #     from sklearn.svm import SVC
    #     # Create the model
    #     mySVC = SVC(gamma='scale', C=10)

    #     # Perform the k-fold cross validation
    #     return 0

    # elif method == 'LogisticRegression':
    #     from sklearn.linear_model import LogisticRegression
    #     # Create the model
    #     myLGR = LogisticRegression(penalty='l2', solver='lbfgs',
    #     multi_class='multinomial')

    #     # Perform the k-fold cross validation
    #     return 0

    # elif method == 'RandomForestClassifier':
    #     from sklearn.ensemble import RandomForestClassifier
    #     # Create the model
    #     myRFC = RandomForestClassifier(max_depth=20, random_state=0,
    #     n_estimators=500)

    #     # Perform the k-fold cross validation
    #     return 0

    # elif method == 'XGBClassifier':
    #     from xgboost import XGBClassifier
    #     # Create the model
    #     myXGB = XGBClassifier(max_depth=5)

    #     # Perform the k-fold cross validation
    #     return 0

    
    return np.array([1]*k)

# my_cross_val('LinearSVC', [0], 3, k=15)

def my_train_test(method, X, y, pi, k):
    X, y = digits.data, digits.target
    # n_train = int(pi * n_samples)
    # n_test = int(n_samples - n_train)
    scores = list()
    # Attempt 1
    # Xn_samples = len(X)
    # X_train = list()
    # Xn_train = pi * Xn_samples
    # X_test = list(X)
    # while len(X_train) < Xn_train:
    #     index = random.randrange(len(X_test))
    #     X_train.append(X_test.pop(index))
    # # print(X_train, X_test)

    # yn_samples = len(y)
    # y_train = list()
    # yn_train = pi * yn_samples
    # y_test = list(y)
    # while len(y_train) < yn_train:
    #     ind = random.randrange(len(y_test))
    #     y_train.append(y_test.pop(index))

    # Attempt 2
    # length = len(X[0])
    # n_train = int(np.ceil(length*pi))
    # n_test = length - n_train

    # perm = np.random.RandomState(1).permutation(length)
    # test_indices = perm[:n_test]
    # train_indices = perm[n_test:]
    # print(test_indices)
    # print(train_indices)
    for i in range(k):

        # Attempt 3
        random.shuffle(X)
        random.shuffle(y)
        train_pct_index = int(pi * len(X))
        X_train, X_test = X[:train_pct_index], X[train_pct_index:]
        y_train, y_test = y[:train_pct_index], y[train_pct_index:]

        if method == 'LinearSVC':
            from sklearn.svm import LinearSVC
            # Create the model
            myLinSVC = LinearSVC(max_iter=5000)
            myLinSVC.fit(X_train, y_train)
            # scores.append(myLinSVC.score(X_test, y_test))
            yhat = myLinSVC.predict(X_test)
            acc = accuracy_score(y_test, yhat)
            scores.append(acc)

        elif method == 'SVC':
            from sklearn.svm import SVC
            # Create the model
            mySVC = SVC(gamma='scale', C=10).fit(X_train, y_train)
            # scores.append(mySVC.score(X_test, y_test))
            yhat = mySVC.predict(X_test)
            acc = accuracy_score(y_test, yhat)
            scores.append(acc)

        elif method == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            # Create the model
            myLGR = LogisticRegression(penalty='l2', solver='lbfgs',
            multi_class='multinomial').fit(X_train, y_train)
            # scores.append(myLGR.score(X_test, y_test))
            yhat = myLGR.predict(X_test)
            acc = accuracy_score(y_test, yhat)
            scores.append(acc)

        elif method == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            # Create the model
            myRFC = RandomForestClassifier(max_depth=20, random_state=0,
            n_estimators=500).fit(X_train, y_train)
            # scores.append(myRFC.score(X_test, y_test))
            yhat = myRFC.predict(X_test)
            acc = accuracy_score(y_test, yhat)
            scores.append(acc)

        elif method == 'XGBClassifier':
            from xgboost import XGBClassifier
            # Create the model
            myXGB = XGBClassifier(max_depth=5).fit(X_train, y_train)
            # scores.append(myXGB.score(X_test, y_test))
            yhat = myXGB.predict(X_test)
            acc = accuracy_score(y_test, yhat)
            scores.append(acc)

        else:
            return np.array([1]*k)
    print(scores)
    return scores

my_train_test('LinearSVC', [0], [1], 0.75, 10)