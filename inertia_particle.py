import numpy as np
from numpy import ndarray
from typing import Callable, Tuple

def f(x: ndarray) -> ndarray:
    return np.zeros_like(x)

def langevin_overdamped(dt, x: ndarray, v: ndarray, D: float, gamma: float, f: Callable, m) -> ndarray:
    """
    Free particle Brownian motion equation resolution.
    dx = f(x)/gamma dt + sqrt(2D dt) * R
    where R is a random number from a normal distribution centered in 0 with std 1
    """
    dim = x.shape[0]
    return - gamma/m * v * dt + f(x) / m * dt + gamma/m * np.sqrt(2 * D * dt) * np.random.normal(0, 1, dim)

def euler_maruyama(D: float, gamma: float,  time: float = 100, dt: float=0.1, m: float = 1, x0: ndarray=np.zeros((1,)), v0: ndarray=np.zeros((1,)), f: Callable = f) -> Tuple[ndarray, ndarray]:
    """
    Resolve the Brownian motion equation for a free particle over a given time.
    """
    if x0.shape != v0.shape:
        print("WARNING: v0 and x0 have different shapes, using v0 = 0 by default")
        v0 = np.zeros_like(x0)
    steps = int(time / dt)
    x = np.zeros((steps, x0.shape[0]))
    v = np.zeros((steps, x0.shape[0]))
    x[0] = x0
    v[0] = v0
    for i in range(1, steps):
        v[i] = v[i-1] + langevin_overdamped(dt, x[i-1], v[i-1], D, gamma, f, m)
        x[i] = x[i-1]  + v[i] * dt
    return x, v

