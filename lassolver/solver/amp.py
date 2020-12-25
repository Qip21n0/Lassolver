import numpy as np
from lassolver.utils.func import *
from lassolver.solver.ista import ISTA

class AMP(ISTA):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)
        self.a = self.M / self.N

    def estimate(self, ite_max=20):
        Onsager = np.zeros((self.M, 1))
        for i in range(ite_max):
            r = self._update_r()
            w = self._update_w(r + Onsager)
            v = self._update_v(r)
            t = self._update_t(v)
            self.s = self._update_s(w, t)
            Onsager = np.sum(self.s != 0) / self.M * (r + Onsager)
            self.mse = self._add_mse()

    def _update_w(self, r):
        return self.s + self.AT @ r

    def _update_v(self, r):
        v = (np.linalg.norm(r)**2 - self.M * self.sigma) / np.trace(self.A2)
        if v < 0:
            v = 1.e-4
        return v

    def _update_t(self, v):
        return v / self.a + self.sigma

    def _update_s(self, w, t):
        return soft_threshold(w, t**0.5)
