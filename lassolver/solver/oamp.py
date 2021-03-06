import numpy as np
from lassolver.utils.func import *
from lassolver.solver.amp import AMP


class OAMP(AMP):
    def __init__(self, A, x, snr, iidG=False):
        super().__init__(A, x, snr)
        self.AAT = A @ A.T
        self.I = np.eye(self.M)
        self.c = (self.N - self.M) / self.M
        self.iidG = iidG

    def estimate(self, T=20, C=2.0, ord='LMMSE'):
        self.W = self.__set_W(1, ord)
        if not self.iidG:
            B = np.eye(self.N) - self.W @ self.A
            self.trW2 = np.trace(self.W @ self.W.T)
            self.trB2 = np.trace(B @ B.T)
        
        for t in range(T):
            r = self._update_r()
            w = self._update_w(r)
            v = self._update_v(r)
            tau = self._update_t(v, ord)
            if t == T-1:
                break
            self.s = self._update_s(C, w, tau)
            self._add_mse()

            if ord == 'LMMSE':
                self.W = self.__set_W(v, ord='LMMSE')
                if not self.iidG:
                    B = np.eye(self.N) - self.W @ self.A
                    self.trW2 = np.trace(self.W @ self.W.T)
                    self.trB2 = np.trace(B @ B.T)
        
        self._output_s(w, t)
        self._add_mse()

    def __set_W(self, v, ord):
        if ord == 'MF':
            W_ = self.AT
        elif ord == 'PINV':
            W_ = np.linalg.pinv(self.A)
        elif ord == 'LMMSE':
            W_ = v * self.AT @ np.linalg.inv(v * self.AAT + self.sigma * self.I)
        else :
            raise NameError("not correct")
        return self.N / np.trace(W_ @ self.A) * W_

    def _update_w(self, r):
        return self.s + self.W @ r

    def _update_t(self, v, ord):
        if self.iidG:
            if ord == 'MF':
                return v / self.a + self.sigma
            elif ord == 'PINV':
                if self.M < self.N:
                    return self.c * v + self.N / (self.N - self.M) * self.sigma
                else :
                    return -self.c * self.sigma
            else :
                tmp = self.c * v + self.sigma
                return (tmp + (tmp**2 + 4 * self.sigma * v)**0.5) / 2
        else :
            return 1/self.N * (self.trB2 * v + self.trW2 * self.sigma)

    def _update_s(self, C, w, tau):
        return C * df(w, tau**0.5)

    def _output_s(self, w, tau):
        return soft_threshold(w, tau**0.5)
