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


def plt_heatmap(confusion_matrix, T):
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
        
        s = confusion_matrix[t][1].astype('int')
        plt.subplot(row, 2, i+1)
        plt.title(f'Quantity (t = {str(t+1)})')
        sns.heatmap(s, cmap="RdBu_r", xticklabels=["x = 0", "x ≠ 0", "sum(diff)"], yticklabels=["diff = 0", "diff ≠ 0", "sum(x)"], annot=True, fmt='d', annot_kws={"fontsize": 12})


def plt_MSE_confusion_matrix(confusion_matrix):
    T = len(confusion_matrix)
    mse_of = {}
    cm = np.array(confusion_matrix)

    mse_of["DOAMP diff!=0"] = cm[:, 0, 0, 2] # diff != 0
    mse_of["DOAMP x=0 & diff!=0"] = cm[:, 0, 0, 0] # TP
    mse_of["DOAMP x!=0 & diff!=0"] = cm[:, 0, 0, 1] # FP

    mse_of["DOAMP diff!=0"] = cm[:, 0, 1, 2] # diff = 0
    mse_of["DOAMP x=0 & diff=0"] = cm[:, 0, 1, 0] # FN
    mse_of["DOAMP x!=0 & diff=0"] = cm[:, 0, 1, 1] # TN

    step = np.arange(0, T+1, 5)

    plt.xlabel("iteration")
    plt.ylabel("MSE")
    plt.xticks(step)
    plt.ylim(1e-4, 1e+1)
    plt.yscale('log')

    for k, v in mse_of.items():
        linestyle = '-'
        color = 'tab:green'

        if 'x=0' in k:
            linestyle = '--'
        elif 'x!=0' in k:
            linestyle = '-.'

        if 'diff=0' in k:
            color = 'tab:red'
        elif 'diff!=0' in k:
            color = 'tab:blue'
        
        plt.plot(v, label=k, linestyle=linestyle, color=color)
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()