import numpy as np
from scipy.stats import norm


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


def xxx(n, rho):
    """
    yobi
    """
    rand = np.random.rand(n)
    cond = soft_threshold(rand-0.5, (1-rho)/2) * True
    for i in range(n):
        if cond[i]:
            q = max(np.abs(rand[i]), np.abs(rand[i]-(1-rho)))
            x[i] = norm.ppf(q/rho, loc=0, scale=rho**(-0.5))
    return x


def onezero(n, p0=0.5):
    """
    generation of binary signal
    """
    return berunouli_gaussian(n, 1-p0) != 0


def bpsk(n):
    """
    grneration of BPSK signal
    """
    signal = onezero(n)
    x = np.ones((n, 1))
    x[signal == 0] = -1
    return x
