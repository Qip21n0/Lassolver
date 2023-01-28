import jax
import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm as normal
from lassolver.utils.func import *
from lassolver.solver.ista import ISTA
import matplotlib.pyplot as plt


class AMP_OPT(ISTA):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)
        self.a = self.M / self.N
        self.v = [None]
        self.tau = [None]

    def estimate(self, T=20):
        def func_mmse(vector, threshold, rho):
            xi = rho**(-1) + threshold
            top = normal.pdf(vector, loc=0, scale=xi**0.5)
            bottom = rho * normal.pdf(vector, loc=0, scale=xi**0.5) + (1-rho) * normal.pdf(vector, loc=0, scale=threshold**0.5)
            return top / bottom * vector
        dfunc_mmse = jax.vmap(jax.grad(func_mmse, argnums=(0)), (0, None, None))
        
        Onsager = np.zeros((self.M, 1))
        for _ in range(T):
            r = self._update_r()
            w = self._update_w(r + Onsager)
            v = self._update_v(r)
            tau = self._update_tau(r + Onsager)
            self.s = self._update_s(w, tau)
            if _ < 3:
                print(self.s)

            rho = np.mean(soft_threshold(w, tau**0.5) != 0)
            Onsager = np.sum(dfunc_mmse(w.reshape((self.N,)), tau**0.5, rho)) / self.M * (r + Onsager)
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
        rho = np.mean(soft_threshold(w, tau**0.5) != 0)
        xi = rho**(-1) + tau
        top = normal.pdf(w, loc=0, scale=xi**0.5) / xi
        bottom = rho * normal.pdf(w, loc=0, scale=xi**0.5) + (1-rho) * normal.pdf(w, loc=0, scale=tau**0.5)
        return np.array(top / bottom * w)

    def result(self):
        super().result()
        ite = np.arange(0, np.shape(self.mse)[0], 1)
        se = np.array([np.log10(v) if v is not None else None for v in self.v])
        plt.scatter(ite, se, c='red')