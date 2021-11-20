#CS 412 Homework 4 Submission Stub
#Zakir Ujjainwala

import numpy as np

def get_splits(n, k):
    all_indices = np.arange(n)
    np.random.shuffle(all_indices)
    split = np.split(all_indices, k)
    
    print(np.asarray(split))
    final = np.asarray(split)

    return final

    # return [[0,2], [1,3]]


def my_cross_val(method, X, y, k):

    return np.array([1]*k)

def my_train_test(method, X, y, pi, k):

    return np.array([1]*k)

get_splits(4,2)