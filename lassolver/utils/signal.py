import jax.numpy as jnp
import numpy as np
from scipy.stats import norm


def bernouli_gaussian(n, rho):
    """
    generation of signal according to Bernouli_Gaussian distribution
    """
    rand = np.random.rand(n)
    x = np.zeros(n)

    for i in range(n):
        if rand[i] < rho/2:
            x[i] = norm.ppf(rand[i]/rho, loc=0, scale=1/rho**0.5)
        elif 1 - rho/2 < rand[i]:
            x[i] = norm.ppf((rand[i] - (1-rho))/rho, loc=0, scale=1/rho**0.5)
        else:
            continue

    return jnp.array(x)


def onezero(n, p1=0.5):
    """
    generation of binary signal
    """
    return 1 * (bernouli_gaussian(n, p1) != 0)


def bpsk(n):
    """
    grneration of BPSK signal
    """
    signal = onezero(n)
    x = np.ones(n)
    x[signal == 0] = -1
    return jnp.array(x)
