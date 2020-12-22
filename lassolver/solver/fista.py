import numpy as np
from lassolver.solver.ista import ISTA

class FISTA(ISTA):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)

    def estimate(self, tau=0.5, ite_max=20):
        L = self.set_lipchitz()
        gamma = 1 / (tau * self.L)
        for i in range(ite_max):
            r = self.update_r(q)
            w = self.update_w(gamma, r)
            s_ = self.s
            self.s = self.update_s(w, 1/L)
            f_ = f
            f = self.update_f(f)
            q = self.update_q(s_)
            self.mse = self.add_mse()

    def update_r(self, q):
        return self.y - self.A @ q

    def update_w(self, q, r, gamma):
        return q + gamma * self.A.T @ r

    def update_f(self, f):
        return (1 + (1 + 4 * f**2)**0.5) / 2

    def update_q(self, f, f_, s_):
        return self.s + (f_ - 1) / f * (self.s - s_)
