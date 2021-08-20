import numpy as np
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *

class damp(dbase):
    def __init__(self, A_p, x, snr, M):
        super().__init__(A_p, x, snr, M)
        self.Onsager_p = np.zeros((self.M_p, 1))

    def update_Onsager(self, s):
        self.s = s
        self.Onsager_p = np.sum(self.s != 0) / self.M * (self.r_p + self.Onsager_p)

    def local_compute(self): 
        self.r_p = self._update_r_p()
        w_p = self._update_w_p()
        r2_p = np.linalg.norm(self.r_p)**2
        return w_p, r2_p

    def _update_r_p(self):
        return self.y - self.A_p @ self.s

    def _update_w_p(self):
        return self.AT_p @ (self.r_p + self.Onsager_p)
        
        
class D_AMP(D_Base):
    def __init__(self, A, x, snr, P):
        super().__init__(A, x, snr, P)
        self.amps = [damp(self.A_p[p], x, snr, self.M) for p in range(self.P)]
        self.sigma = self.__set_sigma()
        self.trA2 = self.__set_trA2()

    def __set_sigma(self):
        sigma = 0
        for p in range(self.P):
            sigma += self.amps[p].sigma_p
        return sigma

    def __set_trA2(self):
        trA2 = 0
        for p in range(self.P):
            trA2 += self.amps[p].trA2_p
        return trA2

    def estimate(self, T=20, log=False):
        lambda_ = np.linspace(3, 2, T)
        w = np.zeros((self.P, self.N, 1))
        self.communication_cost = np.empty(0)

        for t in range(T):
            for p in range(self.P):
                w[p], self.r2[p] = self.amps[p].local_compute()
            w[0] += self.s
            v = self._update_v()
            tau = self._update_tau(v)
            self._update_s(w, lambda_[t], tau, log)

            for p in range(self.P):
                self.amps[p].update_Onsager(self.s)
            self._add_mse()

    def _update_v(self):
        r2 = np.sum(self.r2)
        return (r2 - self.M * self.sigma) / self.trA2

    def _update_tau(self, v):
        return v / self.a + self.sigma

    def _update_s(self, w, lambda_, tau, log):
        s, communication_cost = GCAMP(w, lambda_, tau**0.5, log)
        self.s = s
        self.communication_cost = np.append(self.communication_cost, communication_cost)