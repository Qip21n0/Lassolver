import jax
import numpy as np
from scipy.stats import norm
from jax.scipy.stats import norm as normal


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


def get_parameter(alpha):
    candidates = np.linspace(0, 10, 1_000_000)
    common_part = lambda para: para * norm.pdf(para) - (1 + para**2) * norm.cdf(-para)
    def f(para):
        cp = common_part(para)
        return (alpha + 2 * cp) / (1 + para**2 + 2 * cp)
    i = np.argmax(f(candidates))
    return candidates[i] / alpha**0.5


def GCAMP(w, beta, log=False):
    shita = 0.7
    communication_cost = 0
    P, N, _ = w.shape
    T = beta * shita / (P-1) if P != 1 else 0
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
    F = (U > beta) & (m < (P-1))
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
    F = (U > tau) & (m < (P-1))
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
    z = np.empty(N)
    
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
    F = (U > tau) & (m < (P-1))
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
    Vc = [n for n in range(N) if n not in V]
    for n in Vc:
        b[n] = z[n]
        #b[n] += np.sum([rand(shita * tau_p[p]) for p in range(1, P) if p not in S[n]])
    
    #print(f'|b=w| = {np.sum(np.isclose(b, np.sum(w, axis=0)))}, |V| = {len(V)}, |b=z| = {np.sum(np.isclose(b, z.reshape((N, 1))))}, |Vc| = {len(Vc)}', end=', ')
    #print(f'|w=z| = {np.sum(np.isclose(np.sum(w, axis=0), z.reshape((N, 1))))}, |(b=w) & (b=z)| = {np.sum(np.logical_and(np.isclose(b, np.sum(w, axis=0)), np.isclose(b, z.reshape((N, 1)))))}')
        
    s = u - np.mean(u != 0)*b
    return s.real, communication_cost, np.ravel(U > tau), b, z


def GCOAMP_opt(w, tau_p, log=False):
    shita = 0.7
    tau = np.sum(tau_p)
    communication_cost = 0
    P, N, _ = w.shape
    R = np.zeros((P, N, 1))
    z = np.empty(N)
    
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
    F = (U > tau) & (m < (P-1))
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
    b = np.zeros((N, 1))
    V = np.where(U > tau)[0].tolist()
    for n in V:
        b[n] = np.sum(w[:, n])
    
    #STEP5
    Vc = [n for n in range(N) if n not in V]
    for n in Vc:
        b[n] = z[n]

    rho = np.mean(soft_threshold(b, tau**0.5) != 0)
    def func_mmse(vector, threshold):
        xi = rho**(-1) + threshold
        top = normal.pdf(vector, loc=0, scale=xi**0.5) / xi
        bottom = rho * normal.pdf(vector, loc=0, scale=xi**0.5) + (1-rho) * normal.pdf(vector, loc=0, scale=threshold**0.5)
        return top / bottom * vector

    dfunc_mmse = jax.vmap(jax.grad(func_mmse, argnums=(0)), (0, None))
    reshaped_b = b.reshape(N)
    v_mmse = tau**0.5 * np.mean(dfunc_mmse(reshaped_b, tau))
    C_mmse = tau**0.5 / (tau**0.5 - v_mmse)
    s = C_mmse * (func_mmse(b, tau) - np.mean(dfunc_mmse(reshaped_b, tau)) * b)
    return s.real, communication_cost, np.ravel(U > tau), b, z



def GCOAMP_oracle(zeros, w, tau_p, log=False):
    shita = 0.7
    tau = np.sum(tau_p)
    communication_cost = 0
    P, N, _ = w.shape
    R = np.zeros((P, N, 1))
    z = np.empty(N)
    
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
    for n in range(N):
        z[n] = w[0, n] + np.sum([w[p, n] for p in S[n]])
    F = np.logical_not(zeros) & (m < (P-1))
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
    V = []
    for i, zero in enumerate(zeros):
        if not zero:
            V.append(i)
    for n in V:
        b[n] = np.sum(w[:, n])
        u[n] = soft_threshold(b[n], tau**0.5)
    
    #STEP5
    Vc = [n for n in range(N) if n not in V]
    for n in Vc:
        b[n] = z[n]
        
    s = u - np.mean(u != 0)*b
    return s.real, communication_cost, b, z


def hyperparameter(a):
    f = lambda x: x * norm.pdf(x) - (1 + x**2) * norm.cdf(-x)
    candidates = np.linspace(0, 100, 100000)[1:]
    common = 2 * f(candidates)
    i = np.argmax((a + common) / (1 + candidates**2 + common))
    return candidates[i]