import numpy as np
from numpy import ndarray
from typing import Callable

def f(x: ndarray) -> ndarray:
    return np.zeros_like(x)

def langevin_overdamped(dt, x: ndarray, D: float, gamma: float, f: Callable) -> ndarray:
    """
    Free particle Brownian motion equation resolution.
    dx = f(x)/gamma dt + sqrt(2D dt) * R
    where R is a random number from a normal distribution centered in 0 with std 1
    """
    dim = x.shape[0]
    return f(x) / gamma * dt + np.sqrt(2 * D * dt) * np.random.normal(0, 1, dim)

def euler_maruyama(D: float, gamma: float, time: float = 100, dt: float=0.1, x0: ndarray=np.zeros((1,)), f: Callable = f) -> ndarray:
    """
    Resolve the Brownian motion equation for a free particle over a given time.
    """
    steps = int(time / dt)
    x = np.zeros((steps, x0.shape[0]))
    x[0] = x0
    for i in range(1, steps):
        x[i] = x[i - 1] + langevin_overdamped(dt, x[i - 1], D, gamma, f)
    return x

