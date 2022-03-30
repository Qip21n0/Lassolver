from lassolver.utils.func import *
from lassolver.solver.ista import ISTA
import numpy as np



class AMP(ISTA):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)
        self.normF_A = np.linalg.norm(A, ord='fro')**2
        self.a = self.M / self.N
        self.v = [(np.linalg.norm(self.y)**2 - self.M * self.sigma) / self.normF_A]
        self.tau = []


    def estimate(self, T=20):
        Onsager = jnp.zeros(self.M)
        for _ in range(T):
            self._update_r()
            self._update_w(Onsager)

            self._update_v()
            self._update_tau()

            self.s = self._update_s()
            Onsager = jnp.sum(self.s != 0) / self.M * (self.r + Onsager)

            self._add_mse()
            self.s_history.append(self.s)


    def _update_w(self):
        self.w = self.s + self.AT @ self.r


    def _update_v(self):
        v = (np.linalg.norm(self.r)**2 - self.M * self.sigma) / self.normF_A
        if v < 0:
            v = 1.e-4
        self.v.append(v)


    def _update_tau(self):
        tau = self.v[-1] / self.a + self.sigma
        self.tau.append(tau)


    def _update_s(self):
        self.s = soft_threshold(self.w, self.tau[-1]**0.5)
