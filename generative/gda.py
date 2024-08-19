
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Model:
    phi: float
    mu_pos: Any
    mu_neg: Any
    sigma: Any

def test_train_split(X, y, train_perc=.8):
    indexes = np.random.choice(range(len(X)), size=int(len(X) * train_perc))
    others = [i for i in range(len(X)) if i not in indexes]
    return X[indexes, :], y[indexes], X[others, :], y[others]

def train_model(X, y) -> Model:

    phi = np.mean(y)

    X_pos = X[y == 1]
    X_neg = X[y == 0]

    mu_pos = np.mean(X_pos, axis=0)
    mu_neg = np.mean(X_neg, axis=0)

    cov = np.cov(X, rowvar=False)

    return Model(
        phi, 
        mu_pos, 
        mu_neg,
        cov
    )

def percentage(mu, sigma, x):
    d = x.shape[0]
    det = np.linalg.det(sigma)

    frac = 1 / ((2 * math.pi) ** d/2 * (det ** (1/2)))
    sigma_inv = np.linalg.inv(sigma)
    exp_term = np.exp(-0.5 * np.dot(np.dot((x - mu), sigma_inv), (x - mu).T))
    return frac * exp_term

def test(model: Model, X, y):

    predictions = np.zeros(shape=y.shape)
    for index, x in enumerate(X):
        p_y_zero = 1 - model.phi
        p_x_given_y_zero = percentage(model.mu_neg, model.sigma, x)

        p_y_one = model.phi
        p_x_given_y_one = percentage(model.mu_pos, model.sigma, x)

        if p_y_zero * p_x_given_y_zero > p_y_one * p_x_given_y_one:
            predictions[index] = 0
        else:
            predictions[index] = 1
    
    tp = np.sum((y == 1) & (predictions == 1))
    fp = np.sum((y == 1) & (predictions == 0))
    fn = np.sum((y == 0) & (predictions == 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    import statistics
    f1 = statistics.harmonic_mean([precision, recall])
    return f1


def main():
    data = np.genfromtxt('pima-indians-diabetes.csv', delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]

    num_test_train_splits = 10

    f1s = []
    for run in range(num_test_train_splits):
        print(f"Starting run {run}")
        X_train, y_train, X_test, y_test = test_train_split(X, y)
        model = train_model(X_train, y_train)
        f1s.append(test(model, X_test, y_test))
    
    print(f'Average f1: {sum(f1s) / len(f1s)}')

main()