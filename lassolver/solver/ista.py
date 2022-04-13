from lassolver.utils.func import *
from lassolver.utils.node import *
import jax.numpy as jnp
import numpy as np



class ISTA(Node):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)
        self.r = jnp.zeros(self.M)
        self.w = jnp.zeros(self.N)
        self.s = jnp.zeros(self.N, dtype=jnp.float32)
        self.s_history = []

        self.AT = self.A.T
        self.mse = [None]


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


    def _add_mse(self):
        mse = np.linalg.norm(self.s - self.x)**2 / self.N
        self.mse.append(mse)


    def show_result(self):
        print("final mse: {}".format(self.mse[-1]))
        plt.figure(figsize=(16, 4))

        plt.subplot(121)
        plt.plot(self.x.real)
        plt.plot(self.s.real)
        plt.grid()
        
        plt.subplot(122)
        plt.xlabel('iteration')
        plt.ylabel('MSE[log10]')
        iter = np.shape(self.mse)[0]
        plt.xticks(np.arange(0, iter, 1))
        result = np.array([np.log10(mse) if mse is not None else None for mse in self.mse])
        plt.plot(result)
        plt.grid()