import numpy as np
from lassolver.utils.utils import *

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
