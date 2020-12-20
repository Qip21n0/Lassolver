import numpy as np
from lassolver.utils import *

class AMP(ISTA):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)

    def estimat(self, ite_max=20):
        Onsager = np.zeros((self.N, 1))
        a = self.M / self.N
        for i in range(ite_max):
            r = self.update_r() + Onsager
            w = self.update_w(r)
            v = self.update_v(r)
            t = self.update_t(a, v)
            self.s = self.update_s(w, t)
            Onsager = np.sum(self.s != 0) / self.M * r
            self.add_mse()

    def update_r(self):
        return self.y - self.A @ self.s

    def update_w(self, r):
        return self.s + self.A.T @ r

    def update_v(self, r):
        return (np.linalg.norm(r)**2 - self.M * self.sigma) / np.trace(self.A2)

    def update_t(self, a, v):
        return v / a + self.sigma

    def update_s(self, w, t):
        return soft_threshold(w, t**0.5)

    def add_mse(self):
        super().add_mse()
