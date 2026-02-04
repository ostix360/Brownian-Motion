import numpy as np
from numpy import ndarray

def f(x: ndarray) -> ndarray:
    return np.zeros_like(x)

def diff_equation(dt, x: ndarray, D: float, gamma: float) -> ndarray:
    """
    Free particle Brownian motion equation resolution.
    dx = f(x)/gamma dt + sqrt(2D dt) * R
    where R is a random number from a normal distribution centered in 0 with std 1
    """
    dim = x.shape[0]
    return f(x) / gamma * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, dim)

