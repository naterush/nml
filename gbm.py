# Gradient Boosted Machine
# Mostly written off the wikipedia article. 

from copy import deepcopy
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from nateml.regression_tree import RegressionTree, build_regression_tree
from nateml.regression_tree import predict as predict_submodel
from nateml.split import random_subset, split


@dataclass
class BoostedRegressionTree:
    submodels: List[RegressionTree]
    gammas: List[float]

    def add(self, m, gamma):
        self.submodels.append(m)
        self.gammas.append(gamma)

def predict(model: BoostedRegressionTree, X):
    predictions = np.zeros(X.shape[0])
    for m, gamma in zip(model.submodels, model.gammas):
        predictions += (gamma * predict_submodel(X, m))

    return predictions

def loss(model: BoostedRegressionTree, X, y):
    y_pred = predict(model, X)
    return np.sum(np.square(y - y_pred)) / (2 * y.shape[0])

def read_data():
    data = pd.read_csv('insurance.csv')
    data['sex'] = data['sex'].map(lambda x: 1. if x == 'female' else 0.)
    data['smoker'] = data['smoker'].map(lambda x: 1. if x == 'yes' else 0.)
    encoded = pd.get_dummies(data['region']).astype(float)
    data = data.drop('region', axis=1)
    data = data.join(encoded)
    return data.drop('charges', axis=1).to_numpy(), data['charges'].to_numpy()

def main():
    X, y = read_data()

    X_train, y_train, X_cv, y_cv = split(X, y)

    M = 100
    learning_rate = 0.1  # You can adjust this value
    model = BoostedRegressionTree(
        [
            build_regression_tree(X_train, y_train, max_height=3)
        ], 
        [1]
    )


    for m in range(M):
        X_train_spec, y_train_spec = random_subset(X_train, y_train)

        l = loss(model, X_train_spec, y_train_spec)
        # Compute pseudo-residuals: derivative of the loss function
        # with respect to the aggregated model... specifically for
        # every data point. Make sure it's negative
        y_pred = predict(model, X_train_spec)
        # Loss with respect to X is 1/2n * (y - y_hat)** 2
        # Derivative with respect to y_hat (which is model(x)) is:
        # -1/n * (y - y_hat)
        pseudo_residuals = -(y_train_spec - y_pred) / y_train_spec.shape[0]

        # Train a new model on these pseudo-residuals
        new_tree = build_regression_tree(X_train_spec, pseudo_residuals, max_height=3)
        max_loss_decrease, max_gamma = 0, None
        for gamma in np.arange(-1, 1, 0.001):
            new_temp_model = deepcopy(model)
            new_temp_model.add(new_tree, learning_rate * gamma)

            loss_decrease = l - loss(new_temp_model, X_train_spec, y_train_spec)

            if loss_decrease > max_loss_decrease:
                max_loss_decrease, max_gamma = loss_decrease, gamma

        if max_loss_decrease > 0:
            model.add(new_tree, learning_rate * max_gamma)
            print(f"Iteration {m}, cv loss {round(loss(model, X_cv, y_cv), 0)}, with gamma {learning_rate * max_gamma}")
        else:
            print(f"Iteration {m}, no loss decrease")

    # Def need a cross-validation set to test this on!

main()
