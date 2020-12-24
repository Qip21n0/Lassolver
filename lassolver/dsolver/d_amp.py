import numpy as np
from lassolver.utils.func import *

class amp:
    def __init__(self, A_p, x, snr):
        self.A_p = A_p
        self.d, self.N = A_p.shape
        self.x = x
        Ax_p = A_p @ x
        SNRdB = 10**(-0.1*snr)
        self.sigma_p = np.var(Ax_p) * SNRdB
        n_p = np.random.normal(0, self.sigma_p**0.5, (self.d, 1))
        self.y_p = Ax_p + n_p
        self.s = np.zeros((self.N, 1))
        self.AT_p = A_p.T
        self.trA2_p = np.trace(self.AT_p @ self.A_p)
        self.Onsager_p = np.zeros((self.d, 1))

    def local_compute(self):
        r = self.update_r()
        w = self.update_w(r)
        self.y_As = np.linalg.norm(r)**2

    def update_r(self):
        return self.y_p - self.A_p @ self.s

    def update_w(self, r):
        return self.AT_p @ (r + self.Onsager_p)
        
        
class D_AMP:
    def __init__(self, A, x, snr, P):
        self.M, self.N = A.shape
        self.P = P
        self.a = self.M / self.N
        self.d = int(self.M / self.P)
        self.A_p = A.reshape(P, self.d, self.N)
        self.x = x
        Ax_p = self.set_Ax_p()
        SNRdB = 10**(-0.1*snr)
        self.sigma_p = np.array([np.var(Ax_p[p]) for p in range(P)]) * SNRdB
        self.n_p = np.array([np.random.normal(0, self.sigma_p[p]**0.5, (self.d, 1)) for p in range(P)])
        self.y_p = Ax_p + self.n_p
        self.s = np.zeros((self.N, 1))
        self.mse = np.array([np.linalg.norm(self.s - self.x)**2 / self.N])
        self.AT = self.A.T
        self.A2 = self.AT @ self.A

    def set_Ax_p(self):
        Ax = np.zeros((self.P, self.b, 1))
        for p in range(P):
            Ax[p] = self.A_p[p] @ self.x
        return Ax

    def estimate(self, ite_max=20):
        Onsager_p = np.zeros((self.P, self.d, 1))
        for i in range(ite_max):
            r_p = self.update_r()
            w_p = self.update_w(r + Onsager_p)
            v = self.update_v(r)
            t = self.update_t(v)
            self.s = self.update_s(w, t)
            Onsager_p = np.sum(self.s != 0) / self.M * (r_p + Onsager_p)
            self.mse = self.add_mse()
