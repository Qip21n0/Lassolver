from matplotlib import colors
import numpy as np
from lassolver.utils.func import *
from lassolver.solver.ista import ISTA
import matplotlib.pyplot as plt


class AMP(ISTA):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)
        self.a = self.M / self.N
        self.v = [None]
        self.tau = [None]

    def estimate(self, T=20):
        Onsager = np.zeros((self.M, 1))
        for _ in range(T):
            r = self._update_r()
            w = self._update_w(r + Onsager)
            v = self._update_v(r + Onsager)
            tau = self._update_tau(r + Onsager)
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

    def _update_tau(self, r):
        tau = np.linalg.norm(r)**2 / self.M
        #tau = v / self.a + self.sigma
        self.tau.append(tau)
        return tau

    def _update_s(self, w, tau):
        #tuning_parameter = get_parameter(self.a)
        return soft_threshold(w, tau**0.5)

    def result(self):
        super().result()
        ite = np.arange(0, np.shape(self.mse)[0], 1)
        se = np.array([np.log10(v) if v is not None else None for v in self.v])
        plt.scatter(ite, se, c='red')
