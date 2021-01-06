import numpy as np
from lassolver.utils.func import *

class amp:
    def __init__(self, A, x, snr, M):
        self.A = A
        self.M = M
        self.Mp, self.N = A.shape
        self.x = x
        Ax = A @ x
        SNRdB = 10**(-0.1*snr)
        self.sigma = np.var(Ax) * SNRdB
        n = np.random.normal(0, self.sigma_p**0.5, (self.Mp, 1))
        self.y = Ax + n
        self.s = np.zeros((self.N, 1))
        self.AT = A.T
        self.trA2 = np.trace(self.AT @ self.A)
        self.Onsager = np.zeros((self.Mp, 1))

    def receive_s(self, s):
        self.s = s

    def local_compute(self):
        r = self._update_r()
        w = self._update_w(r)
        y_As = np.linalg.norm(r)**2
        self.Onsager = np.sum(self.s != 0) / self.M * (r + self.Onsager)
        return w, y_As

    def _update_r(self):
        return self.y - self.A @ self.s

    def _update_w(self, r):
        return self.AT @ (r + self.Onsager)
        
        
class D_AMP:
    def __init__(self, A, x, snr, P):
        self.M, self.N = A.shape
        self.P = P
        self.a = self.M / self.N
        self.Mp = int(self.M / self.P)
        self.A_p = A.reshape(P, self.Mp, self.N)
        self.x = x
        self.amps = [amp(self.A_p[p], x, snr, self.M) for p in range(P)]
        self.sigma = self.set_sigma()
        self.s = np.zeros((self.N, 1))
        self.mse = np.array([np.linalg.norm(self.s - self.x)**2 / self.N])
        self.trA2 = self.set_trA2()
        self.y_As_p = np.zeros(self.P)

    def set_sigma(self):
        sigma = 0
        for p in range(self.P):
            sigma += self.amps[p].sigma
        return sigma / self.P

    def set_tr2(self):
        trA2 = 0
        for p in range(self.P):
            trA2 += self.amps[p].trA2
        return trA2

    def estimate(self, ite_max=20):
        for i in range(ite_max):
            for p in range(P):
                self.amp[p].receive(self.s)
                w_p[p], self.y_As_p[p] = self.amps[p].local_compute()
            w_p[0] += self.s
            v = self._update_v()
            t = self._update_t(v)
            self.s = self._update_s(w_p, t)
            self.mse = self.add_mse()

    def _update_v(self):
        y_As = np.sum(self.y_As_p)
        return (y_As - self.M * self.sigma) / self.trA2

    def _update_t(self, v):
        return v / self.a + self.sigma

    def _update_s(self, w, t):
        return GCAMP(w, t**0.5)
