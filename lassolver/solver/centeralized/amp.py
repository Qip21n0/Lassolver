import numpy as np

class AMP(ISTA):
    def __init__(self, A, x, n):
        super().__init__(A, x, n)
        self.sigma = np.var(n)
        self.Onsager = np.zeros((self.N, 1))

    def estimat(self, ite_max=20):
        Onsager = np.zeros((self.N, 1))
        a = self.M / self.N
        for i in range(ite_max):
            r = self.update_r(Onsager)
            w = self.update_w(r)
            v = self.update_v()
            t = self.update_t(a, v)
            self.s = self.update_s(w, t)
            Onsager = np.sum(self.s != 0) / self.M * r
            mse_ = np.linalg.norm(self.s - self.x)**2 / self.N
            self.mse = np.append(self.mse, mse_)

    def update_r(self):
        return self.y - self.A @ self.s + Onsager

    def update_w(self, r):
        return self.s + self.A @ r

    def update_v(self):
        return (np.linalg.norm(self.y - self.A @ self.s)**2 - self.M * self.sigma) / np.trace(self.A2)

    def update_t(self, a, v):
        return v / a + self.sigma


    def update_s(self, w, t):
        return soft_threshold(w, t**0.5) 
