import numpy as np
import matplotlib.pyplot as plt
from lassolver.utils.func import *

class ISTA:
    def __init__(self, A, x, noise):
        self.A = A
        self.M, self.N = A.shape
        self.K = np.sum(x != 0)
        self.x = x
        Ax = A @ x
        if type(noise) is int:
            SNRdB = 10**(0.1*noise) / self.P
            self.sigma = np.linalg.norm(Ax)**2 / SNRdB
            self.n = np.random.normal(0, self.sigma**0.5, (self.M, 1))
        elif type(noise).__module__ == 'numpy':
            self.sigma = np.var(noise)
            self.n = noise.copy()
        else :
            raise ValueError
        self.y = Ax + self.n
        self.s = np.zeros((self.N, 1))
        self.mse = np.array([None])
        self.AT = A.T
        self.A2 = self.AT @ self.A
        self.zero_index = x == 0
        self.non_zero_index = x != 0
        self.mse_zero = np.array([None])
        self.mse_non_zero = np.array([None])
        self.mse_hist_bins = []

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

        sum_4_zero = 0
        sum_4_non_zero = 0
        for k, v in enumerate(self.zero_index):
            if v:
                sum_4_zero += self._square_error_4_component(k)
            elif not v:
                sum_4_non_zero += self._square_error_4_component(k)
            else:
                raise ValueError("Not Correct Value")
        self.mse_zero = np.append(self.mse_zero, sum_4_zero[0] / (self.N - self.K))
        self.mse_non_zero = np.append(self.mse_non_zero, sum_4_non_zero[0] / self.K)

    def _square_error_4_component(self, i):
        return (self.s[i] - self.x[i])**2

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
        result = np.array([np.log10(mse) if mse is not None else None for mse in self.mse])
        plt.plot(result)
        plt.grid()