import numpy as np
from scipy.stats import truncnorm, norm


def soft_threshold(r, gamma):
    """
    soft-thresholding function
    """
    return np.maximum(np.abs(r) - gamma, 0.0) * np.sign(r)


def df(r, gamma):
    """
    divergence-free function
    """
    eta = soft_threshold(r, gamma)
    return eta - np.mean(eta != 0) * r


def GCAMP(w, beta, log=False):
    shita = 0.7
    communication_cost = 0
    P, N, _ = w.shape
    T = beta * shita / (P-1)
    R = np.zeros((P, N, 1))
    z = np.zeros((N, 1))
    
    #STEP1
    for p in range(1, P):
        R[p] = np.abs(w[p]) > T
        candidate = np.where(R[p])[0]
        for n in candidate:
            communication_cost += 1
            send_to1(n, w[p, n])
    
    #STEP2
    S = [np.where(R[:, n])[0] for n in range(N)]
    m = np.sum(R, axis=0)
    U = np.empty((N, 1))
    for n in range(N):
        upper = (P - 1 - m[n]) * T
        z[n] = w[0, n] + np.sum([w[p, n] for p in S[n]])
        U[n] = np.abs(z[n]) + upper
    F = (U > beta) * (m < (P-1))
    candidate = np.where(F)[0]
    for n in candidate:
        communication_cost += 1
        broadcast_others(n)
    
    #STEP3
    F_R = F * np.logical_not(R)
    for p in range(1, P):
        #print("p: {}".format(p))
        candidate = np.where(F_R[p])[0]
        for n in candidate:
            communication_cost += 1
            send_to1(n ,w[p, n])
    if log: 
        print("Rp: {} \t F: {} \t F\\Rp: {}".format(np.sum(R), np.sum(F), np.sum(F_R)-np.sum(F)))
        print("Total Communication Cost: {}".format(communication_cost))
        print("="*50)
    
    #STEP4
    s = np.zeros((N, 1))
    b = np.zeros((N, 1))
    V = np.where(U > beta)[0].tolist()
    for n in V:
        b[n] = np.sum(w[:, n])
        s[n] = soft_threshold(b[n], beta)
    
    return s.real, communication_cost


def GCAMP_exp(w, tau_p, log=False):
    shita = 0.7
    tau = np.sum(tau_p)
    communication_cost = 0
    P, N, _ = w.shape
    R = np.zeros((P, N, 1))
    
    #STEP1
    for p in range(1, P):
        R[p] = np.square(w[p]) > tau_p[p] * shita
        candidate = np.where(R[p])[0]
        for n in candidate:
            communication_cost += 1
            send_to1(n, w[p, n])
    
    #STEP2
    S = [np.where(R[:, n])[0] for n in range(N)]
    m = np.sum(R, axis=0)
    U = np.empty((N, 1))
    for n in range(N):
        upper = np.sum([tau_p[p] for p in range(1, P) if p not in S[p]])
        U[n] = (w[0, n] + np.sum(w[p, n] for p in S[n]))**2 + upper * shita
    F = (U > tau) * (m < (P-1))
    candidate = np.where(F)[0]
    for n in candidate:
        communication_cost += 1
        broadcast_others(n)
    
    #STEP3
    F_R = F * np.logical_not(R)
    for p in range(1, P):
        #print("p: {}".format(p))
        candidate = np.where(F_R[p])[0]
        for n in candidate:
            communication_cost += 1
            send_to1(n ,w[p, n])
    if log: 
        print("Rp: {} \t F: {} \t F\\Rp: {}".format(np.sum(R), np.sum(F), np.sum(F_R)-np.sum(F)))
        print("Total Communication Cost: {}".format(communication_cost))
        print("="*50)

    #STEP4
    s = np.zeros((N, 1))
    V = np.where(U > tau)[0].tolist()
    for n in V:
        w_sum = np.sum(w[:, n])
        s[n] = soft_threshold(w_sum, tau**0.5)
    return s.real, communication_cost


def send_to1(n, w):
    #print("n: {}, w: {}".format(n, w))
    pass


def broadcast_others(n):
    #print("n: {}".format(n))
    pass


def GCOAMP(w, tau_p, log=False):
    shita = 0.7
    tau = np.sum(tau_p)
    communication_cost = 0
    P, N, _ = w.shape
    R = np.zeros((P, N, 1))
    z = [0] * N
    
    #STEP1
    for p in range(1, P):
        R[p] = np.square(w[p]) > tau_p[p] * shita
        candidate = np.where(R[p])[0]
        for n in candidate:
            communication_cost += 1
            send_to1(n, w[p, n])
    
    #STEP2
    S = [np.where(R[:, n])[0] for n in range(N)]
    m = np.sum(R, axis=0)
    U = np.empty((N, 1))
    for n in range(N):
        upper = np.sum([tau_p[p] for p in range(1, P) if p not in S[p]])
        z[n] = w[0, n] + np.sum([w[p, n] for p in S[n]])
        U[n] = z[n]**2 + upper * shita
    F = (U > tau) * (m < (P-1))
    candidate = np.where(F)[0]
    for n in candidate:
        communication_cost += 1
        broadcast_others(n)
    
    #STEP3
    F_R = F * np.logical_not(R)
    for p in range(1, P):
        #print("p: {}".format(p))
        candidate = np.where(F_R[p])[0]
        for n in candidate:
            communication_cost += 1
            send_to1(n ,w[p, n])
    if log: 
        print("Rp: {} \t F: {} \t F\\Rp: {}".format(np.sum(R), np.sum(F), np.sum(F_R)-np.sum(F)))
        print("Total Communication Cost: {}".format(communication_cost))
        print("="*50)
    
    #STEP4
    u = np.zeros((N, 1))
    b = np.zeros((N, 1))
    V = np.where(U > tau)[0].tolist()
    for n in V:
        b[n] = np.sum(w[:, n])
        u[n] = soft_threshold(b[n], tau**0.5)
    
    #STEP5
    #if approx: rand = beta * truncnorm.rvs(-1, 1, loc=0, scale=1, size=N-K)
    #else : rand = Rrandom(u, beta, K)#(tau - tau_p[0])**0.5 * truncnorm.rvs(-1, 1, loc=0, scale=1, size=N-K)
    Vc = [n for n in range(N) if n not in V]
    for n in Vc:
        b[n] = z[n]
        #b[n] += np.sum([rand(shita * tau_p[p]) for p in range(1, P) if p not in S[n]])
        
    s = u - np.mean(u != 0)*b
    return s.real, communication_cost, b - np.sum(w, axis=0)


def rand(tau):
    return tau**0.5 * truncnorm.rvs(-1, 1, loc=0, scale=1, size=1)


def Rrandom(u, t, K):
    N = u.shape[0]
    
    u0 = np.histogram(u, bins=N)
    Pu = u0[0]/N
    Pu = np.append(Pu, 0)
    u1 = u0[1]

    phi = lambda x: norm.pdf((x-u1)/t)/t

    maxu = np.argmax(Pu)
    phi_x = phi(u1[maxu])
    max = np.max(np.sum(Pu * phi_x))
    rand = np.empty(N-K)

    for i in range(N-K):
        while True:
            x, y = np.random.rand(2)
            a = -t + 2*t*x
            phi_a = phi(a)
            A = np.sum(Pu * phi_a)
            if max*y <= A:
                rand[i] = a
                break
    return rand