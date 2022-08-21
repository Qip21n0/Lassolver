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
        sns.heatmap(s, cmap="RdBu_r", xticklabels=["x = 0", "x ≠ 0", "sum(diff)"], yticklabels=["diff ≠ 0", "diff = 0", "sum(x)"], annot=True, fmt='d', annot_kws={"fontsize": 12})


def plt_MSE_confusion_matrix(confusion_matrix):
    T = len(confusion_matrix)
    mse_of = {}
    cm = np.array(confusion_matrix)

    mse_of["DOAMP diff!=0"] = cm[:, 0, 0, 2] # diff != 0
    mse_of["DOAMP x=0 & diff!=0"] = cm[:, 0, 0, 0] # TP
    mse_of["DOAMP x!=0 & diff!=0"] = cm[:, 0, 0, 1] # FP

    mse_of["DOAMP diff=0"] = cm[:, 0, 1, 2] # diff = 0
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


def plt_ratio_confusion_matrix(confusion_matrix):
    T = len(confusion_matrix)
    ratio_of = {}
    cm = np.array(confusion_matrix)
    N = cm[0, 1, 2, 2]

    ratio_of["DOAMP diff!=0"] = cm[:, 1, 0, 2] / N # diff != 0
    ratio_of["DOAMP x=0 & diff!=0"] = cm[:, 1, 0, 0] / N # TP
    ratio_of["DOAMP x!=0 & diff!=0"] = cm[:, 1, 0, 1] / N # FP

    ratio_of["DOAMP diff=0"] = cm[:, 1, 1, 2] / N # diff = 0
    ratio_of["DOAMP x=0 & diff=0"] = cm[:, 1, 1, 0] / N # FN
    ratio_of["DOAMP x!=0 & diff=0"] = cm[:, 1, 1, 1] / N # TN

    xstep = np.arange(0, T+1, 5)
    ystep = np.arange(0, 1.1, 0.1)

    plt.xlabel("iteration")
    plt.ylabel("Quantity")
    plt.xticks(xstep)
    plt.yticks(ystep)
    plt.ylim(0, 1.1)

    for k, v in ratio_of.items():
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

    plt.legend()
    plt.grid()


def plt_evaluation_index(evaluation_index):
    plt.ylim(0, 1)
    ystep = np.arange(0, 1.1, 0.1)
    plt.yticks(ystep)

    for k, v in evaluation_index.items():
        plt.plot(v, label=k)
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()


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
        hist, bins = np.histogram(s_without_None, bins=100)
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


def plt_w_z_history(target, type, T):
    wbz = target.w_b_z_history
    tau = np.array(target.tau[1:])**0.5
    length = len(wbz[0]["TP"][0])

    n = T//11
    flag = False
    if n * 11 < T:
        n += 1
        flag = True

    plt.figure(figsize=(20, 6*n))
    for i in range(n+1):
        t = 11 * i - 1 if i != 0 else 0
        if i == n and flag:
            t = -1
        w = wbz[t][type][0]
        z  = wbz[t][type][2]
        
        plt.subplot(n+1, 2, 2*i+1)
        plt.title(f'w & z (t = {str(t+1)})')
        plt.scatter(np.arange(length), w, label=type+' w', color='tab:blue')
        plt.scatter(np.arange(length), z, label=type+' z', color='tab:orange')
        plt.plot(np.array([tau[t]]*length), color='black', linestyle='dashed')
        plt.plot(np.array([-tau[t]]*length), color='black', linestyle='dashed')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.grid()

        j = np.logical_not(np.isnan(w))
        hist_w, bins = np.histogram(w[j], bins=50)
        hist_z, _ = np.histogram(z[j], bins=bins)
        max = np.max([*hist_w, *hist_z])
        
        plt.subplot(n+1, 2, 2*(i+1))
        plt.title(f'histgram (t = {str(t+1)})')
        plt.hist([w, z], bins=bins, label=[type+' w', type+' z'])
        plt.vlines(tau[t], 0, max, colors='black', linestyles='dashed')
        plt.vlines(-tau[t], 0, max, colors='black', linestyles='dashed')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.grid()