import numpy as np


def split_data(X, y, test_size=.2, random_seed=42):
    np.random.seed(random_seed)
    indexes = np.random.choice(range(len(X)), size=int(len(X) * (1 - test_size)))
    others = [i for i in range(len(X)) if i not in indexes]
    return X[indexes, :], y[indexes], X[others, :], y[others]

def standard_scaler(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def main():
    data = np.genfromtxt('generative/pima-indians-diabetes.csv', delimiter=",")

    X = standard_scaler(data[:, :-1])
    y = data[:, -1]

    # Split into train and test sets
    X_train, y_train, X_test, y_test = split_data(X, y)

    
main()



