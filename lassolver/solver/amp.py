import numpy as np
from lassolver.utils.func import *
from lassolver.solver.ista import ISTA


class AMP(ISTA):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)
        self.a = self.M / self.N
        self.v = []
        self.tau = []

    def estimate(self, T=20):
        Onsager = np.zeros((self.M, 1))
        for _ in range(T):
            r = self._update_r()
            w = self._update_w(r + Onsager)
            v = self._update_v(r)
            tau = self._update_tau(v)
            self.s = self._update_s(w, tau)
            Onsager = np.sum(self.s != 0) / self.M * (r + Onsager)
            self._add_mse()

    def _update_w(self, r):
        return self.s + self.AT @ r

    def _update_v(self, r):
        v = (np.linalg.norm(r)**2 - self.M * self.sigma) / np.trace(self.A2)
        if v < 0:
            v = 1.e-4
        self.v.append(v)
        return v

    def _update_tau(self, v):
        tau = v / self.a + self.sigma
        self.tau.append(tau)
        return tau

    def _update_s(self, w, tau):
        return soft_threshold(w, tau**0.5)
