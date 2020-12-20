import numpy as np
from lassolver.utils.utils import *
from lassolver.solver.centeralized.amp import AMP

class OAMP(AMP):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)
        self.AAT = A @ A.T
        self.I = np.eye(self.M)

    def estimate(self, C=1.75, ord='LMMSE', ite_max=20):
        self.AT = self.A.T
        self.W = self.set_W(1, ord)
        a = self.M / self.N
        c = (self.N - self.M) / self.M
        for i in range(ite_max):
            r = self.update_r()
            w = self.update_w(r)
            v = self.update_v(r)
            t = self.update_t(a, c, v, ord)
            self.s = self.update_s(C, w, t) if i != ite_max-1 else self.output_s(w, t)
            self.mse = self.add_mse()
            if ord == 'LMMSE':
                self.W = self.set_W(v, ord='LMMSE')

    def set_W(self, v, ord):
        if ord == 'MF':
            W_ = self.AT
        elif ord == 'PINV':
            W_ = np.linalg.pinv(self.A)
        elif ord == 'LMMSE':
            W_ = v * self.AT @ np.linalg.inv(v * self.AAT + self.sigma * self.I)
        else :
            raise NameError("not correct")
        return self.N / np.trace(W_ @ self.A) * W_

    def update_w(self, r):
        return self.s + self.W @ r

    def update_t(self, a, c, v, ord):
        if ord == 'MF':
            return v / a + self.sigma
        elif ord == 'PINV':
            if self.M < self.N:
                return c * v + self.N / (self.N - self.M) * self.sigma
            else :
                return -c * self.sigma
        else :
            tmp = c * v + self.sigma
            return (tmp + (tmp**2 + 4 * self.sigma * v)**0.5) / 2

    def update_s(self, C, w, t):
        return C * DF(w, t**0.5)

    def output_s(self, w, t):
        return soft_threshold(w, t**0.5)
