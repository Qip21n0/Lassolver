import numpy as np
from scipy.stats import norm
from lassolver.utils.func import *


def bernouli_gaussian(n, rho):
    """
    generation of signal according to Bernouli_Gaussian distribution
    """
    rand = np.random.rand(n)
    x = np.zeros((n, 1))
    for i in range(n):
        if rand[i] < rho/2:
            x[i] = norm.ppf(rand[i]/rho, loc=0, scale=1/rho**0.5)
        elif 1 - rho/2 < rand[i]:
            x[i] = norm.ppf((rand[i] - (1-rho))/rho, loc=0, scale=1/rho**0.5)
    return x


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
    x = np.ones((n, 1))
    x[signal == 0] = -1
    return x
