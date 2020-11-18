# util file
from scipy.stats import norm

import numpy as np


def random_gen(n, rho):
    """
    ガウス・ベルヌーイ分布に従う原信号の生成
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


def special_gaussian(Lambda):
    """
    ガウス分布の確率密度関数(p.d.f)と累積分布関数(c.d.f)を組み合わせた関数
    """
    return (1 + Lambda**2) * norm.cdf(Lambda) - Lambda * norm.pdf(Lambda)


def set_threshold(alpha, sigma):
    """
    ISTAで使う閾値の設定
    """
    Lambda_opt = 0
    para_max = 0
    for Lambda in np.arange(0, 5*sigma**0.5, 1/(100*5*sigma**0.5)) :
        Gaussian = special_gaussian(Lambda)
        with np.errstate(divide='ignore'):  # 0除算のRuntimeWarningを無視
            para = (1 - 2/alpha * Gaussian) / (1 + Lambda**2 - 2 * Gaussian)
        if para_max < para:
            Lambda_opt = Lambda
    return Lambda_opt/alpha**0.5


def soft_threshold(r, gamma):
    """
    軟判定閾値関数
    """
    return np.maximum(np.abs(r) - gamma, 0.0) * np.sign(r)


def d_soft_threshold(r, gamma):
    """
    軟判定閾値関数の導関数
    """
    return np.array(np.abs(r) >= gamma, dtype=np.int)


def DF(r, gamma):
    """
    divergence-freeな関数
    """
    return soft_threshold(r, gamma) - np.mean(d_soft_threshold(r, gamma), axis=None) * r