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

    v = cc.copy() / standard if standard != 0 else 0
    v = np.append(None, v)
    plt.plot(v, label=label, color=color, linestyle=linestyle)
    plt.legend(loc="lower right")


def plt_MSE(mse, label, T, color=None, linestyle=None):
    step = np.arange(0, T+1, 5)

    plt.xlabel("iteration")
    plt.ylabel("MSE")
    plt.xticks(step)
    plt.ylim(1e-4, 1e+1)
    plt.yscale('log')

    plt.plot(mse, label=label, color=color, linestyle=linestyle)
    plt.legend()


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


def plt_SE(se, T, color=None):
    step = np.arange(0, T+1, 1)
    plt.scatter(step, se, c=color)


def plt_MSE_and_hist(target, T, limit=False):
    n = T//11
    flag = False
    if n * 11 < T:
        n += 1
        flag = True

    plt.figure(figsize=(14, 6*n))
    for i in range(n+1):
        t = 11 * i - 1 if i != 0 else 0
        if i == n and flag:
            t = -1
        mse_all, hist_all, bins_all = target.mse_hist_bins[t][0].copy()
        mse_zero, hist_zero, bins_zero = target.mse_hist_bins[t][1].copy()
        mse_non_zero, hist_non_zero, bins_non_zero = target.mse_hist_bins[t][2].copy()

        plt.subplot(n+1, 2, 2*i+1)
        plt.title(f'MSE (t = {str(t+1)})')
        plt.plot(bins_all, mse_all, label="all")
        plt.plot(bins_zero, mse_zero, label="x = 0")
        plt.plot(bins_non_zero, mse_non_zero, label="x != 0")
        if limit:
            upper = np.max(mse_all)
            plt.ylim(None, upper * 1.1)
        plt.legend()
        plt.grid()

        plt.subplot(n+1, 2, 2*(i+1))
        plt.title(f'Quantity (t = {str(t+1)})')
        plt.plot(bins_all, hist_all, label="all")
        plt.plot(bins_zero, hist_zero, label="x = 0")
        plt.plot(bins_non_zero, hist_non_zero, label="x != 0")
        plt.legend()
        plt.grid()


def plt_s_diff_non_zero(target, T):
    n = T//11
    flag = False
    if n * 11 < T:
        n += 1
        flag = True

    plt.figure(figsize=(14, 6*n))
    for i in range(n+1):
        t = 11 * i - 1 if i != 0 else 0
        if i == n and flag:
            t = -1
        s = target.s_history_4_diff_non_zero[t].real.copy()
        hist, bins = np.histogram(s, bins=50)
        hist = np.append(hist, 0)

        plt.subplot(n+1, 2, 2*i+1)
        plt.title(f's (t = {str(t+1)})')
        plt.plot(target.x)
        plt.plot(s)
        plt.grid()

        plt.subplot(n+1, 2, 2*(i+1))
        plt.title(f's histogram (t = {str(t+1)})')
        plt.plot(bins, hist)
        plt.grid()