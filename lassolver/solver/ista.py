from lassolver.utils.func import *
from lassolver.utils.node import *
import jax.numpy as jnp
import numpy as np



class ISTA(Node):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)
        self.r = jnp.zeros(self.M)
        self.w = jnp.zeros(self.N)


    def estimate(self, T=20, tau=0.5):
        L = self.__set_lipchitz()
        gamma = 1 / (tau * L)
        for _ in range(T):
            self._update_r()
            self._update_w(gamma)
            self._update_s(1/L)

            self._add_mse()
            self.s_history.append(self.s)


    def __set_lipchitz(self):
        w, _ = np.linalg.eig(self.A2)
        return np.max(w)


    def _update_r(self):
        self.r = self.y - self.A @ self.s


    def _update_w(self, gamma):
        self.w = self.s + gamma * self.AT @ self.r


    def _update_s(self, gamma):
        self.s = soft_threshold(self.w, gamma)