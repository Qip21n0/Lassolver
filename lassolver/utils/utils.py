from cProfile import label
import time
from functools import wraps
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
        plt.yscale('log')
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
        s_without_None = np.array([])
        for s_comp in s:
            if s_comp is not None:
                s_without_None = np.append(s_without_None, s_comp)
        hist, bins = np.histogram(s_without_None, bins=50)
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


def plt_heatmap(impact_table, T):
    n = T//11
    flag = False
    if n * 11 < T:
        n += 1
        flag = True

    row = n//2 + 1

    plt.figure(figsize=(14, 6*row))
    for i in range(n+1):
        t = 11 * i - 1 if i != 0 else 0
        if i == n and flag:
            t = -1
        
        s = impact_table[t][1].astype('int')
        plt.subplot(row, 2, i+1)
        plt.title(f'Quantity (t = {str(t+1)})')
        sns.heatmap(s, cmap="RdBu_r", xticklabels=["x = 0", "x ≠ 0", "sum(diff)"], yticklabels=["diff = 0", "diff ≠ 0", "sum(x)"], annot=True, fmt='d', annot_kws={"fontsize": 12})


def plt_MSE_impact_table(impact_table):
    T = len(impact_table)
    mse_of = {}
    impact_table = np.array(impact_table)

    mse_of["x=0 & diff=0"] = impact_table[:, 0, 0, 0]
    mse_of["x!=0 & diff=0"] = impact_table[:, 0, 0, 1]
    mse_of["x=0 & diff!=0"] = impact_table[:, 0, 1, 0]
    mse_of["x=!0 & diff!=0"] = impact_table[:, 0, 1, 1]

    mse_of["diff=0"] = impact_table[:, 0, 0, 2]
    mse_of["diff!=0"] = impact_table[:, 0, 1, 2]
    mse_of["x=0"] = impact_table[:, 0, 2, 0]
    mse_of["x!=0"] = impact_table[:, 0, 2, 1]

    mse_of["all"] = impact_table[:, 0, 2, 2]

    step = np.arange(0, T+1, 5)
    dot = [i for i in range(T) if i % 5 == 0]
    if dot[-1] != T-1:
        dot.append(T-1)

    plt.xlabel("iteration")
    plt.ylabel("MSE")
    plt.xticks(step)
    plt.ylim(1e-4, 1e+1)
    plt.yscale('log')

    for k, v in mse_of.items():
        linestyle = '-'
        marker = None
        mkv = None
        if 'x=0' in k:
            linestyle = '--'
        elif 'x!=' in k:
            linestyle = '-.'

        if 'diff=0' in k:
            marker = 'o'
            mkv = dot
        elif 'diff!=0' in k:
            marker = '*'
            mkv = dot
        
        plt.plot(v, label=k, linestyle=linestyle, marker=marker, markevery=mkv)
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()