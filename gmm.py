# Gaussian Mixture Model, let's go

import math
from typing import Any, List
import numpy as np
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

@dataclass
class Model:
    phis: List[float]
    mus: List[Any]
    covs: List[Any]

def matrix_product(m1, m2, opp):

    if m1.shape[1] != m2.shape[0]:
        raise ValueError('Error, shapes dont match')
    
    result = np.zeros((m1.shape[0], m2.shape[1]))

    for i in range(m1.shape[0]):
        for j in range(m2.shape[1]):
            result[i, j] = opp(m1[i, :], m2[:, j])

    return result


def distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))


def closest_points(centers, all_points):
    # Can I do this entirely vectorized? So just make a matrix of C x P size
    # and then compute the distance in each of these

    distance_matrix = matrix_product(
        all_points, 
        centers.T,
        distance
    )

    return np.argmin(distance_matrix, axis=1)

def get_starting_gmm_model(points, k):
    # We initialize an empty GMM model simply by picking
    # k random points, and then using these to derive
    # both the phis, the mus, and the covariance matrixes
    starting_point_indexes = np.random.choice(points.shape[0], k)
    starting_points = points[starting_point_indexes]

    closest = closest_points(starting_points, points)

    phis = []
    mus = []
    covs = []
    for i in range(k):
        mask = closest == i
        points_for_center = points[mask]

        phi = np.sum(mask) / points.shape[0]
        mu = np.average(points_for_center, axis=0)

        mean_normalized_points = points_for_center - mu

        cov = np.matmul(mean_normalized_points.T, mean_normalized_points) / (np.sum(mask) - 1)

        phis.append(phi)
        mus.append(mu)
        covs.append(cov)

    return Model(phis, mus, covs)


def multivariate_gaussian_pdf(mu, cov, x):

    n = x.shape[0]

    det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)

    frac = 1 / (((2 * math.pi)**(n / 2)) * (det ** .5))
    exp = np.exp(-.5 * np.matmul((x - mu), np.matmul(cov_inv, (x - mu).T)))
    return frac * exp


def get_responsibilities(model, points, k):
    
    responsibilities = np.zeros((points.shape[0], k))

    for j in range(k):
        phi = model.phis[j]
        mu = model.mus[j]
        cov = model.covs[j]

        for i, x in enumerate(points):
            prob_x_given_z = multivariate_gaussian_pdf(mu, cov, x)
            p_z = phi

            prob_x = sum(
                phi * multivariate_gaussian_pdf(mu1, cov1, x) for phi, mu1, cov1 in zip(model.phis, model.mus, model.covs)
            )

            p_z_given_x = (prob_x_given_z * p_z) / prob_x

            responsibilities[i, j] = p_z_given_x

    return responsibilities

def plot(model, points):

    # Plot the results
    plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
    plt.title("2D Gaussian Distribution Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')

    for mu, cov in zip(model.mus, model.covs):
        plt.scatter(mu[0], mu[1], color='red', s=100, marker='x')
        
        # Plot the covariance ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = patches.Ellipse(
            xy=mu, width=width, height=height,
            angle=angle, fill=False, color='black'
        )
        plt.gca().add_patch(ellipse)


    plt.show()

def calculate_likelyhood(model, points):
    total = 0
    for p in points:
        subtotal = 0
        for phi, mu, cov in zip(model.phis, model.mus, model.covs):
            p_z = phi
            p_x_given_z = multivariate_gaussian_pdf(mu, cov, p)
            subtotal += p_x_given_z * p_z
        total += np.log(subtotal)
    
    return total

def build_gmm_model(points, k=2, iters=1000):
    starting_model = get_starting_gmm_model(points, k)
    print(f'On iteration {0}, likelyhood={calculate_likelyhood(starting_model, points)}')

    n = points.shape[0]

    model = starting_model
    for iter_number in range(iters):
        # n x k matrix. where the rows sum to one
        responsibilities = get_responsibilities(model, points, k)

        # Update all the phis
        phis = []
        mus = []
        covs = []
        for j in range(k):
            w_j_sum = np.sum(responsibilities[:, j])
            phi = 1 / n * w_j_sum
            mu = np.sum(points * responsibilities[:, j].reshape(points.shape[0], -1), axis=0) / w_j_sum

            mean_adjusted = points - mu
            cov = np.matmul(mean_adjusted.T, mean_adjusted * responsibilities[:, j].reshape(-1, 1)) / w_j_sum

            phis.append(phi)
            mus.append(mu)
            covs.append(cov)

        model = Model(phis, mus, covs)

        if iter_number % 50 == 0:
            plot(model, points)
            print(f'On iteration {iter_number}, likelyhood={calculate_likelyhood(model, points)}')

    return model


def main():
    np.set_printoptions(suppress=True)

    np.random.seed(40)

    samples1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 2]], 100)
    samples2 = np.random.multivariate_normal([4, 4], [[1, 0.5], [0.5, 2]], 100)
    samples3 = np.random.multivariate_normal([10, 10], [[1, 0.5], [0.5, 2]], 100)

    data = np.vstack((samples1, samples2, samples3))
    model = build_gmm_model(data, k=3)

    plot(model, data)


main()