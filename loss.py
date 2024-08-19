



import numpy as np
import matplotlib.pyplot as plt


def mse(predictions, y):
    return np.sum(np.square(predictions - y)) / (2 * predictions.shape[0])


def plot_loss(loss_array, title="Loss over iterations", xlabel="Iterations", ylabel="Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_array)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()