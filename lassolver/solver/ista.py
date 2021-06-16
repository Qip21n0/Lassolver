import numpy as np
import matplotlib.pyplot as plt
from lassolver.utils.func import *

class ISTA:
    def __init__(self, A, x, snr):
        self.A = A
        self.M, self.N = A.shape
        self.x = x
        Ax = A @ x
        SNRdB = 10**(-0.1*snr)
        self.sigma = np.var(Ax) * SNRdB
        self.n = np.random.normal(0, self.sigma**0.5, (self.M, 1))
        self.y = Ax + self.n
        self.s = np.zeros((self.N, 1))
        self.mse = np.array([np.linalg.norm(self.s - self.x)**2 / self.N])
        self.AT = A.T
        self.A2 = self.AT @ self.A

    def estimate(self, T=20, tau=0.5):
        L = self.__set_lipchitz()
        gamma = 1 / (tau * L)
        for _ in range(T):
            r = self._update_r()
            w = self._update_w(gamma, r)
            self.s = self._update_s(w, 1/L)
            self.mse = self._add_mse()

    def __set_lipchitz(self):
        w, _ = np.linalg.eig(self.A2)
        return np.max(w)

    def _update_r(self):
        return self.y - self.A @ self.s

    def _update_w(self, gamma, r):
        return self.s + gamma * self.AT @ r

    def _update_s(self, w, thre):
        return soft_threshold(w, thre)

    def _add_mse(self):
        mse = np.linalg.norm(self.s - self.x)**2 / self.N
        self.mse = np.append(self.mse, mse)

    def result(self):
        print("final mse: {}".format(self.mse[-1]))

        plt.figure(figsize=(16, 4))
        plt.subplot(121)
        plt.plot(self.x.real)
        plt.plot(self.s.real)
        plt.grid()

        plt.subplot(122)
        plt.xlabel('iteration')
        plt.ylabel('MSE[log10]')
        ite = np.shape(self.mse)[0]
        plt.xticks(np.arange(0, ite, 1))
        plt.plot(np.log10(self.mse))
        plt.grid()