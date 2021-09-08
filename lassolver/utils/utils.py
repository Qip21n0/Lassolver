import time
from functools import wraps
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt


def plt_CC(cc, label, T, N, P, color=None, linestyle=None):
    x_step = np.arange(0, T+1, 5)
    y_step = np.arange(0, 1.1, 0.1)
    standard = N * (P-1)

    plt.xlabel("iteration")
    plt.ylabel("Communication Cost")
    plt.xticks(x_step)
    plt.yticks(y_step)
    plt.ylim(0, 1.1)

    v = cc.copy() / standard
    v = np.append(None, v)
    plt.plot(v, label=label, color=color, linestyle=linestyle)
    plt.legend(loc="lower right")
    plt.grid()


def plt_MSE(mse, label, T, color=None, linestyle=None):
    step = np.arange(0, T+1, 5)

    plt.xlabel("iteration")
    plt.ylabel("MSE")
    plt.xticks(step)
    plt.ylim(1e-3, 1e+1)
    plt.yscale('log')

    plt.plot(mse, label=label, color=color, linestyle=linestyle)
    plt.legend()
    plt.grid()


def plt_MSE_cond(mse, label, sim, color=None):
    ite = np.arange(0, sim, 1)
    s = mse.astype(np.double)
    m = np.isfinite(s)

    plt.xlabel('condition number κ')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.yscale('log')

    plt.plot(ite[m], s[m], label=label, color=color)
    plt.legend()
    plt.grid()