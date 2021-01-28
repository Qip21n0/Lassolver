import numpy as np
from scipy.stats import truncnorm


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
    T = beta * shita / (P - 1)  
    #STEP1
    for p in range(P-1):
        R[p+1] = np.array(np.abs(w[p+1]) > T, dtype=np.bool)
        for n in range(N):
            send_to1(n, w[p+1, n])
    #STEP2
    S = [np.where(R[:, n])[0] for n in range(N)]
    m = np.sum(R, axis=0)
    U = (P - 1 -m) * T
    for n in range(N):
        U[n] += np.abs(w[0, n] + np.sum([w[p, n] for p in S[n]]))
    F = np.array(U > beta, dtype=np.bool) * np.array(m < (P-1), dtype=np.bool)
    for n in range(N):
        if F[n]:
            broadcast_others(n)
    #STEP3
    F_Rp = F * np.logical_not(R)
    for p in range(P-1):
        #print("p: {}".format(p+1))
        for n in range(N):
            if F_Rp[p+1, n]:
                send_to1(n ,w[p+1, n])
    #STEP4
    s = np.zeros((N, 1))
    V = np.array(U > beta, dtype=np.bool)
    for n in range(N):
        if V[n]:
            w_sum = np.sum(w[:, n])
            s[n] = soft_threshold(w_sum, beta)
    return s.real


def send_to1(n, w):
    #print("n: {}, w: {}".format(n, w))
    pass


def broadcast_others(n):
    #print("n: {}".format(n))
    pass


def GCOAMP(w, beta, shita=0.8):
    P, N, _ = w.shape
    R = np.zeros((P, N, 1))
    T = beta * shita / (P - 1)
    #STEP1
    for p in range(P-1):
        R[p+1] = np.array(np.abs(w[p+1]) > T, dtype=np.bool)
        for n in range(N):
            send_to1(n, w[p+1, n])
    #STEP2
    S = [np.where(R[:, n])[0] for n in range(N)]
    m = np.sum(R, axis=0)
    U = (P - 1 -m) * T
    for n in range(N):
        U[n] += np.abs(w[0, n] + np.sum([w[p, n] for p in S[n]]))
    F = np.array(U > beta, dtype=np.bool) * np.array(m < (P-1), dtype=np.bool)
    for n in range(N):
        if F[n]:
            broadcast_others(n)
    #STEP3
    F_Rp = F * np.logical_not(R)
    for p in range(P-1):
        #print("p: {}".format(p+1))
        for n in range(N):
            if F_Rp[p+1, n]:
                send_to1(n ,w[p+1, n])
    #STEP4
    u = np.zeros((N, 1))
    m = np.zeros((N, 1))
    V = np.array(U > beta, dtype=np.bool)
    for n in range(N):
        if V[n]:
            m[n] = np.sum(w[:, n])
            u[n] = soft_threshold(m[n], beta)
    #STEP5
    K = np.sum(u != 0)
    rand = truncnorm.rvs(-1, 1, loc=0, scale=1, size=N-K)
    num = np.random.choice(N-K, N-K, replace=False)
    cnt = 0
    for n in range(N):
        if not V[n]:
            m[n] = beta * rand[num[cnt]]
            cnt += 1
    s = u - np.mean(u != 0)*m
    return s.real