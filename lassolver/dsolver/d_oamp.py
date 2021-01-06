import numpy as np
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *

class oamp(base):
    def __init__(self, A, x, snr, M):
        super().__init__(A, x, snr, M)

    def receive_s(self, s):
        self.s = s

    def local_compute(self, W):
        self.r = self._update_r()
        w = self._update_w(W)
        y_As = np.linalg.norm(self.r)**2
        return w, y_As

    def _update_r(self):
        return self.y - self.A @ self.s

    def _update_w(self, W):
        return W @ self.r


class D_OAMP(D_Base):
    def __init__(self, A, x, snr, P):
        super().__init__(A, x, snr, P)
        self.AAT = A @ A.T
        self.I = np.eye(self.M)
        self.c = (self.N - self.M) / self.M
        self.oamps = [oamp(self.A_p[p], x, snr, self.M) for p in range(P)]
        self.sigma = self.set_sigma()
        self.trA2 = self.set_trA2()

    def set_sigma(self):
        sigma = 0
        for p in range(self.P):
            sigma += self.oamps[p].sigma
        return sigma / self.P

    def set_trA2(self):
        trA2 = 0
        for p in range(self.P):
            trA2 += self.oamps[p].trA2
        return trA2

    def estimate(self, C=1.75, ord='LMMSE', ite_max=20):
        self.W = self.__set_W(1, ord)
        w_p = np.zeros((self.P, self.N, 1))
        for i in range(ite_max):
            for p in range(self.P):
                w_p[p], self.y_As_p[p] = self.oamps[p].local_compute()
            w_p[0] += self.s
            v = self._update_v()
            t = self._update_t(v)
            self.s = self._update_s(w_p, t)
            for p in range(self.P):
                self.oamps[p].update_Onsager(self.s)
            self.mse = self._add_mse()

    def _update_v(self):
        y_As = np.sum(self.y_As_p)
        return (y_As - self.M * self.sigma) / self.trA2

    def _update_t(self, v):
        return v / self.a + self.sigma

    def _update_s(self, w, t):
        return GCAMP(w, t**0.5)