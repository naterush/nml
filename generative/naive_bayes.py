
import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class Model:
    phi_y: float
    dist_pos: Dict[int, Dict[int, float]]
    dist_neg: Dict[int, Dict[int, float]]

def test_train_split(X, y, train_perc=.8):
    indexes = np.random.choice(range(len(X)), size=int(len(X) * train_perc))
    others = [i for i in range(len(X)) if i not in indexes]
    return X[indexes, :], y[indexes], X[others, :], y[others]

def calculate_feature_percentages(X):
    total_observations = X.shape[0]
    feature_percentages = {}
    for feature_index, feature in enumerate(X.T):
        values, counts = np.unique(feature, return_counts=True)
        percentages = counts / total_observations
        values_to_percentages = dict(zip(values, percentages))
        feature_percentages[feature_index] = values_to_percentages
    return feature_percentages

def train_model(X, y) -> Model:

    phi_y = np.mean(y)

    # Split into positive and negative, and calculate the percentage
    # of each class with these
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    dist_pos = calculate_feature_percentages(X_pos)
    dist_neg = calculate_feature_percentages(X_neg)

    return Model(
        phi_y, 
        dist_pos,
        dist_neg
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

        p_x_given_y_pos = np.prod(
            [model.dist_pos[feature_index].get(value, 0) for feature_index, value in enumerate(x)]
        ) 
        p_x_given_y_neg = np.prod(
            [model.dist_neg[feature_index].get(value, 0) for feature_index, value in enumerate(x)]
        )

        p_x = (p_x_given_y_pos * model.phi_y) + (p_x_given_y_neg * (1 - model.phi_y))

        p_y_pos = (p_x_given_y_pos * model.phi_y) / p_x
        p_y_neg = (p_x_given_y_neg * model.phi_y) / p_x

        if p_y_neg > p_y_pos:
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


def clean(data):
    """
    preg = Number of times pregnant
    plas = Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    pres = Diastolic blood pressure (mm Hg)
    skin = Triceps skin fold thickness (mm)
    test = 2-Hour serum insulin (mu U/ml)
    mass = Body mass index (weight in kg/(height in m)^2)
    pedi = Diabetes pedigree function
    age = Age (years)
    class = Class variable (1:tested positive for diabetes, 0: tested negative for diabetes)
    """

    # plas, we do chunks of 10
    data[:, 1] = data[:, 1] // 10
    # pres, chunks of 5
    data[:, 2] = data[:, 2] // 10
    # skin, chunks of 5
    data[:, 3] = data[:, 3] // 5
    # test, chunks of 100
    data[:, 4] = data[:, 4] // 100
    # mass, chunks of 5
    data[:, 5] = data[:, 5] // 5
    # pedi, chunks of .1
    data[:, 6] = data[:, 6] // .1
    # age, chunks of 5
    data[:, 7] = data[:, 7] // 5

    return data.astype(int)






def main():
    data = np.genfromtxt('pima-indians-diabetes.csv', delimiter=",")
    data = clean(data)
    X = data[:, :-1]
    y = data[:, -1]

    num_test_train_splits = 10

    f1s = []
    for run in range(num_test_train_splits):
        print(f"Starting run {run}")
        X_train, y_train, X_test, y_test = test_train_split(X, y)
        model = train_model(X_train, y_train)
        f1s.append(test(model, X_test, y_test))

    print(F'Average f1s={sum(f1s) / len(f1s)}')

main()