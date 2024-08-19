import numpy as np

def random_subset(*args, size=.5):
    arg = args[0]
    indexes = np.random.random_integers(0, arg.shape[0] - 1, int(arg.shape[0] * size))
    return (arg[indexes] for arg in args)


def split(X, y, cv_percentage=.2):
    # Gives you back train, test, cross-validation
    cv_indexes = np.random.random_integers(0, X.shape[0] - 1, int(X.shape[0] * cv_percentage))
    X_cv, y_cv = X[cv_indexes], y[cv_indexes]

    train_indexes = [i for i in range(X.shape[0]) if i not in cv_indexes]
    X_train, y_train = X[train_indexes], y[train_indexes]
    
    return X_train, y_train, X_cv, y_cv