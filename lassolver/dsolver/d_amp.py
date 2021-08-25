import numpy as np
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *

class damp(dbase):
    def __init__(self, A_p, x, snr, M):
        super().__init__(A_p, x, snr, M)
        self.Onsager_p = np.zeros((self.M_p, 1))

    def receive_trA2(self, trA2):
        self.trA2 = trA2

    def update_Onsager(self, s):
        self.s = s
        self.Onsager_p = np.sum(self.s != 0) / self.M * (self.r_p + self.Onsager_p)

    def local_compute(self): 
        self.r_p = self._update_r_p()
        w_p = self._update_w_p()
        r2_p = np.linalg.norm(self.r_p)**2
        v_p = (np.linalg.norm(self.r_p)**2 - self.M * self.sigma_p) / self.trA2
        if v_p < 0: v_p = 1e-5
        tau_p = self.N / self.M * v_p + self.sigma_p
        return w_p, v_p, tau_p

    def _update_r_p(self):
        return self.y - self.A_p @ self.s

    def _update_w_p(self):
        return self.s / self.P + self.AT_p @ (self.r_p + self.Onsager_p)
        
        
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
        w = np.zeros((self.P, self.N, 1))
        self.communication_cost = np.empty(0)

        for p in range(self.P):
            self.amps[p].receive_trA2(self.trA2)

        for t in range(T):
            for p in range(self.P):
                w[p], self.v[p], self.tau[p] = self.amps[p].local_compute()
            #w[0] += self.s
            v = self._update_v()
            tau = self._update_tau()
            self.www.append([w.copy(), self.tau.copy()])
            if log: print("{}/{}: tau = {}, v = {}".format(t+1, T, tau, v))
            self._update_s(w, log)

            for p in range(self.P):
                self.amps[p].update_Onsager(self.s)
            self._add_mse()

    def _update_v(self):
        #r2 = np.sum(self.r2)
        #v = (r2 - self.M * self.sigma) / self.trA2
        #return v if v > 0 else 1e-4
        return np.sum(self.v)

    def _update_tau(self):
        #return v / self.a + self.sigma
        return np.sum(self.tau)

    def _update_s(self, w, log):
        s, communication_cost = GCAMP(w, self.tau, log)
        self.s = s
        self.communication_cost = np.append(self.communication_cost, communication_cost)