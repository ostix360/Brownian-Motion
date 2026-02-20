import numpy as np
from numpy import ndarray
from typing import Callable

def f(x: ndarray, v: ndarray, dt: float) -> ndarray:
    """
    Force that acts on the particle
    x: ndarray of shape (n_walkers, 3) 
    v: ndarray of shape (n_walkers, 3)
    dt: float
    :return: ndarray of shape (n_walkers, 3) 
    """
    return np.zeros_like(x)

def langevin_inertia(dt, x: ndarray, v: ndarray, D: float, m:float, gamma: float, f: Callable,) -> ndarray:
    """
    Free particle Brownian motion equation resolution.
    dx = f(x)/gamma dt + sqrt(2D dt) * R
    where R is a random number from a normal distribution centered in 0 with std 1
    """
    dim = x.shape
    return - gamma/m * v * dt + f(x, v, dt) / m * dt + gamma/m * np.sqrt(2 * D * dt) * np.random.normal(0, 1, dim)


def euler_maruyama_3d(D: float, gamma: float, time: float = 100, dt: float=0.1, m: float = 1, var: float=1, x_0: ndarray=np.zeros((1, 3)), f: Callable = f,) -> ndarray:
    """
    Resolve the Brownian motion equation for 2 free particles linked by a spring
    :return x: ndarray of shape (steps, n_walkers, 3) where x[:, :, i] with i in [x, y, z] is the position in the 3d space
    """
    steps = int(time / dt)
    x = np.zeros((steps, x_0.shape[0], 3))
    x[0, :, :] = x_0
    v = np.zeros((steps, x_0.shape[0], 3))
    for i in range(1, steps):
        v[i] = v[i - 1, :, :] + langevin_inertia(dt, x[i - 1, :, :], v[i - 1, :, :], D, m, gamma, f)
        x[i] = x[i-1]  + v[i] * dt
    return x, v
