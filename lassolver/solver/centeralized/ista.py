import numpy as np
from lassolver.utils.utils import *

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
        self.A2 = A.T @ A

    def estimate(self, tau=0.5, ite_max=20):
        self.AT = self.A.T
        self.L = self.set_lipchitz()
        gamma = 1 / (tau * self.L)
        for i in range(ite_max):
            r = self.update_r()
            w = self.update_w(gamma, r)
            self.s = self.update_s(w, 1/self.L)
            self.mse = self.add_mse()

    def set_lipchitz(self):
        w, _ = np.linalg.eig(self.A2)
        return np.max(w)

    def update_r(self):
        return self.y - self.A @ self.s

    def update_w(self, gamma, r):
        return self.s + gamma * self.AT @ r

    def update_s(self, w, thre):
        return soft_threshold(w, thre)

    def add_mse(self):
        mse = np.linalg.norm(self.s - self.x)**2 / self.N
        return np.append(self.mse, mse)
