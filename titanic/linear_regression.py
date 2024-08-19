
import numpy as np
from typing import Any, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class LogisticRegression:
    theta: Any


def build_regression_model(X, y, num_iters=100000, alpha=.01):

    np.random.seed(42)
    theta = np.random.rand(1, X.shape[1]) - 1
    for iteration in range(num_iters):
        cost = compute_cost(X, y, theta)

        # Thus, the gradient descent step is:
        adjusted_X = (predict(X, theta) - y.reshape(-1, 1)) * X
        gradient = 1 / adjusted_X.shape[0] * np.sum(adjusted_X, axis=0)
        theta = theta - alpha * gradient

        if iteration % 500 == 0:
            print(f'On iteration: {iteration}, cost: {cost}')
    
    return LogisticRegression(theta)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    raw = sigmoid(np.matmul(X, theta.T).astype('float'))
    return np.clip(raw, 1e-7, 1 - 1e-7)

def compute_cost(X, y, theta):
    # theta is 1 x m parameters
    # X is n X m. Predictions is thus n X 1, like y
    predictions = predict(X, theta)

    y = y.reshape(-1, 1)

    # Binary cross-entropy loss is 
    # 1/n * Sum over i - y_i * log(y_hat_i) - (1 - y_i) * log(1 - y_hat_i)
    return np.sum(- y * np.log(predictions) - (1 - y) * np.log(1 - predictions)) / y.shape[0]

def gradient_descent(X, y, theta, alpha, num_iters):
    for iteration in range(num_iters):
        # Thus, the gradient descent step is:
        adjusted_X = (predict(X, theta) - y) * X
        gradient = 1 / adjusted_X.shape[0] * np.sum(adjusted_X, axis=1)
        theta = theta - alpha * gradient

    return theta





def get_training_data():
    df = pd.read_csv('train.csv')
    data = df.to_numpy()
    return np.delete(data, 1, axis=1), data[:, 1].astype('float')

def get_test_data():
    df = pd.read_csv('test.csv')
    data = df.to_numpy()
    return data

def one_hot_encode(X, feature):
    col = X[:, feature]
    X = np.delete(X, feature, axis=1)

    uniques = np.unique(col)
    new_columns = []
    for u in uniques:
        new_column = np.zeros(col.shape, dtype='int')
        new_column[col == u] = 1
        new_columns.append(new_column)

    return new_columns


def to_binary(X, feature):
    col = X[:, feature]
    X = np.delete(X, feature, axis=1)
    uniques = np.unique(col)

    if len(uniques) > 2:
        raise ValueError(f'Cannot encode to binary, not enough values in feature {feature}: {uniques}')

    new_column = np.zeros(col.shape, dtype='int')
    new_column[col == uniques[0]] = 0
    new_column[col == uniques[1]] = 1

    return new_column

def fill_nan_with_first(X, feature):
    col = X[:, feature]
    col[np.isnan(col)] = col[0] # bug
    return col

def clean_data(X):
    """
    Columns are: PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

    PassengerId - ignore, just for prediction
    Pclass - one hot encode this; ticket class
    Name - ignore this for now
    Sex - to binary
    Age - leave as a number
    SibSp - leave as a number
    Parch - leave as a number
    Ticket - ignore this for now
    Fare - leave as a number
    Cabin - ignore this for now
    Embarked - one hot encode this
    """
    # We keep track of the columns, and then build this into an array
    # without changing X, so we can reference variables by their original
    # index

    feature_columns = []

    # 0 - PassengerId - ignore, just for prediction

    # 1 - Pclass - one hot encode this; ticket class
    feature_columns.extend(one_hot_encode(X, 1))

    # 2 - Name - ignore this for now

    # 3 - Sex - to binary
    feature_columns.append(to_binary(X, 3))

    # 4 - Age - leave as a number
    X[:, 4][X[:, 4]!=X[:, 4]] = 0
    feature_columns.append(X[:, 4])

    # 5 - SibSp - leave as a number
    feature_columns.append(X[:, 5])
    
    # 6 - Parch - leave as a number
    feature_columns.append(X[:, 6])

    # 7 - Ticket - ignore this for now
    
    # 8 - Fare - leave as a number
    feature_columns.append(X[:, 8])
    
    # 9 - Cabin - ignore this for now
    
    # 10 - Embarked - one hot encode this. First, 
    # fill nans
    X[:, 10] = np.where([x is np.nan for x in X[:, 10]], 'S', X[:, 10])
    feature_columns.extend(one_hot_encode(X, 10))

    return np.array(feature_columns).T

def main():
    

    X, y = get_training_data()
    X = clean_data(X)[0:10, :]
    y = y[0:10]

    model = build_regression_model(X, y)
    print(model)

    # Test:
    test_data = get_test_data()
    X = clean_data(test_data)

    predictions = predict(X, model.theta)
    predictions = np.where(predictions > .5, 1, 0)
    print(predictions)

    result = pd.DataFrame({'PassengerId': test_data[:, 0], 'Survived': predictions[:, 0]})
    result.to_csv('logistic_regression_result.csv', index=False)

main()
