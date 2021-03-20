import numpy as np
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *

class amp(base):
    def __init__(self, A, x, snr, M):
        super().__init__(A, x, snr, M)
        self.Onsager = np.zeros((self.Mp, 1))

    def update_Onsager(self, s):
        self.s = s
        self.Onsager = np.sum(self.s != 0) / self.M * (self.r + self.Onsager)

    def local_compute(self): 
        self.r = self._update_r()
        w = self._update_w()
        y_As = np.linalg.norm(self.r)**2
        return w, y_As

    def _update_r(self):
        return self.y - self.A @ self.s

    def _update_w(self):
        return self.AT @ (self.r + self.Onsager)
        
        
class D_AMP(D_Base):
    def __init__(self, A, x, snr, P):
        super().__init__(A, x, snr, P)
        self.amps = [amp(self.A_p[p], x, snr, self.M) for p in range(P)]
        self.sigma = self.set_sigma()
        self.trA2 = self.set_trA2()

    def set_sigma(self):
        sigma = 0
        for p in range(self.P):
            sigma += self.amps[p].sigma
        return sigma / self.P

    def set_trA2(self):
        trA2 = 0
        for p in range(self.P):
            trA2 += self.amps[p].trA2
        return trA2

    def estimate(self, ite_max=20):
        w_p = np.zeros((self.P, self.N, 1))
        self.communication_cost = np.empty(0)
        for i in range(ite_max):
            for p in range(self.P):
                w_p[p], self.y_As_p[p] = self.amps[p].local_compute()
            w_p[0] += self.s
            v = self._update_v()
            t = self._update_t(v)
            self._update_s(w_p, t)
            for p in range(self.P):
                self.amps[p].update_Onsager(self.s)
            self.mse = self._add_mse()

    def _update_v(self):
        y_As = np.sum(self.y_As_p)
        return (y_As - self.M * self.sigma) / self.trA2

    def _update_t(self, v):
        return v / self.a + self.sigma

    def _update_s(self, w, t):
        s, communication_cost = GCAMP(w, t**0.5)
        self.s = s
        self.communication_cost = np.append(self.communication_cost, communication_cost)