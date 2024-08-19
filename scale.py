

import numpy as np


def standard_scaler(X):
    # If it's one dimensional
    if len(X.shape) == 1:
        uniques = np.unique(X)
        if len(uniques) > 2:
            return (X - np.mean(X)) / np.std(X)
        
    for col in range(X.shape[1]):
        uniques = np.unique(X[:, col])
        if len(uniques) > 2:
            X[:, col] = (X[:, col] - np.mean(X[:, col])) / np.std(X[:, col])
        
    return X

