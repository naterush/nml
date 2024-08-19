import numpy as np


def main():

    samples = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 2]], 100)
    cov = np.matmul(samples.T, samples) / (samples.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    biggest = np.argmax(eigenvalues)
    pc = eigenvectors[biggest]

    print(eigenvalues / np.sum(eigenvalues))

    # This is the explained variance ratio, insanely enough.
    print(np.cumsum(np.sort(eigenvalues / np.sum(eigenvalues))[::-1]))

    # projectect feautres
    new_samples = np.matmul(samples, pc.T)
    #print(new_samples)

main()
