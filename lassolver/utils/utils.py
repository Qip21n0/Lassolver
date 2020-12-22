import numpy as np
from scipy.stats import norm


def bernouli_gaussian(n, rho):
    """
    generation of signal according to Berouli-Gaussian distribution
    """
    rand = np.random.rand(n)
    x = np.zeros((n, 1))
    for i in range(n):
        if rho/2 <= rand[i]:
            if rand[i] <= 1 - rho/2 :
                x[i] = 0
            else:
                x[i] = norm.ppf((rand[i] - (1-rho))/rho, loc=0, scale=1/rho**0.5)
        else:
            x[i] = norm.ppf(rand[i]/rho, loc=0, scale= 1/rho**0.5)
    return x


def soft_threshold(r, gamma):
    """
    soft-thresholding function
    """
    return np.maximum(np.abs(r) - gamma, 0.0) * np.sign(r)


def DF(r, gamma):
    """
    divergence-free function
    """
    eta = soft_threshold(r, gamma)
    return eta - np.mean(eta != 0) * r

def GCAMP(w, beta, shita=0.8):
    P, N, _ = w.shape
    R = np.zeros((P, N, 1))
