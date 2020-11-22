import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #a kernel density estimation method with a Gaussian kernel with standard deviation h (2 pts) in kde.
    # Compute the number of samples created
    N = len(samples)

    # Create a linearly spaced vector
    pos = np.arange(-5, 5.0, 0.1)

    # Estimate the density from the samples using a kernel density estimator
    norm = np.sqrt(2 * np.pi) * h * N
    res = np.sum(np.exp(-(pos[np.newaxis, :] - samples[:, np.newaxis]) ** 2 / (2 * h ** 2)), axis=0) / norm

    # Form the output variable
    estDensity = np.stack((pos, res), axis=1)

    return estDensity
