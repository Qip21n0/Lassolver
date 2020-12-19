import numpy as np

class ISTA:
    def __init__(self, A, x, n):
        self.A = A
        self.M, self.N = A.shape
        self.x = x
        self.n = n
        self.y = A @ x + n
        self.s = np.zeros((self.N, 1))
        self.mse = np.array(np.linalg.norm(self.s - self.x)**2 / self.N)
        self.A2 = A.T @ A

    def estimate(self, ite_max=20):
        self.L = self.set_lipchitz()
        gamma = 1 / (self.tau * self.L)
        for i in range(ite_max):
            r = self.update_r()
            w = self.update_w(gamma)
            self.s = self.update_s()
            mse_ = np.linalg.norm(self.s - self.x)**2 / self.N
            self.mse = np.append(self.mse, mse_)

    def set_lipchitz(self):
        w, _ = np.linalg.eig(self.A2)
        return np.max(w)

    def update_r(self):
        return self.y - self.A @ self.s

    def update_w(self, gamma):
        return self.s + gamma * self.A.T @ self.r

    def update_s(self):
        return soft_threshold(self.w, 1/self.L)
