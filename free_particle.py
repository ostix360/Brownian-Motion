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

def euler_maruyama_spring(D: float, gamma: float, time: float = 100, dt: float=0.1, x0: ndarray=np.zeros((1,)), x1: ndarray=1.5*np.ones((1,)), f: Callable = f, k: float=0.1, l0: float =1) -> ndarray:
    """
    Resolve the Brownian motion equation for 2 free particles linked by a spring
    :return x: ndarray of shape (steps, n_walkers, 2) where x[:, :, 0] is the position of the first particle and x[:, :, 1] is the position of the second particle
    """
    steps = int(time / dt)
    x = np.zeros((steps, x0.shape[0], 2))
    x[0,:, 0] = x0
    x[0,:, 1] = x1
    for i in range(1, steps):
        dist = x[i,:,1] - x[i, :, 0]
        if dist > 0:
            dist = x[i,:,1] - x[i, :, 0]
            x[i, :, 0] = x[i - 1, :, 0] + langevin_overdamped(dt, x[i - 1, :, 0], D, gamma, f) - k * (dist - l0) * dt
            x[i, :, 1] = x[i - 1, :, 1] + langevin_overdamped(dt, x[i - 1, :, 1], D, gamma, f) + k * (dist - l0) * dt
        else:
            dist = x[i,:,0] - x[i, :, 1]
            x[i, :, 0] = x[i - 1, :, 0] + k * (dist - l0) * dt + langevin_overdamped(dt, x[i - 1, :, 0], D, gamma, f)
            x[i, :, 1] = x[i - 1, :, 1] - k * (dist - l0) * dt + langevin_overdamped(dt, x[i - 1, :, 1], D, gamma, f)
    return x
