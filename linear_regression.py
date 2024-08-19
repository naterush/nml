# Linear Regression: just about the simplest thing
# https://en.wikipedia.org/wiki/Linear_regression is probably good enough

from dataclasses import dataclass
from typing import Any

import numpy as np

from nateml.loss import mse, plot_loss
from nateml.scale import standard_scaler

@dataclass
class LinearRegressionModel:
    w: Any
    b: Any

def predict(model: LinearRegressionModel, X):
    return np.matmul(X, model.w) + model.b

def train_linear_regression(X, y, num_iters=100000, learning_rate=.00003):

    X = standard_scaler(X)
    w = np.zeros(X.shape[1])
    b = 0
    model = LinearRegressionModel(w, b)
    losses = []

    for iteration in range(num_iters):
        predictions = predict(model, X)
        loss = mse(predictions, y)

        # Ok, loss function is 1/2n X sum of examples of (y_i - y_hat_i) ** 2
        # Want derivative with respect to w. 
        # y_hat_i = wx + b
        # 1 / 2n X 2 * (y_i - y_hat_i) * - x

        # Weight updates
        errors = predictions - y
        w_gradient = (np.dot(X.T, errors) / X.shape[0])
        b_gradient = np.mean(errors)

        model.w = model.w - learning_rate * w_gradient
        model.b = model.b - learning_rate * b_gradient

        losses.append(loss)

        if iteration % 50 == 0:
            print(f'Iteration {iteration}, loss {loss}')

    plot_loss(losses)
        
    return model






    

