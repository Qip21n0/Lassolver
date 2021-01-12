import numpy as np
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *
from lassolver.utils.utils import *

class oamp(base):
    def __init__(self, A, x, snr, M):
        super().__init__(A, x, snr, M)

    def receive_s(self, s):
        self.s = s

    def receive_W(self, W):
        self.W = W.T

    def local_compute(self):
        self.r = self._update_r()
        w = self._update_w()
        y_As = np.linalg.norm(self.r)**2
        return w, y_As

    def _update_r(self):
        return self.y - self.A @ self.s

    def _update_w(self):
        return self.W @ self.r


class D_OAMP(D_Base):
    def __init__(self, A, x, snr, P, iidG=False):
        super().__init__(A, x, snr, P)
        self.A = A
        self.AT = A.T
        self.AAT = A @ A.T
        self.I = np.eye(self.M)
        self.c = (self.N - self.M) / self.M
        self.oamps = [oamp(self.A_p[p], x, snr, self.M) for p in range(P)]
        self.sigma = self.set_sigma()
        self.trA2 = self.set_trA2()
        self.iidG = iidG

    def set_sigma(self):
        sigma = 0
        for p in range(self.P):
            sigma += self.oamps[p].sigma
        return sigma / self.P

    def set_trA2(self):
        trA2 = 0
        for p in range(self.P):
            trA2 += self.oamps[p].trA2
        return trA2

    def estimate(self, C=2.0, ord='LMMSE', ite_max=20):
        self.W = self.__set_W(1, ord)
        self.W_p = self.W.T.reshape(self.P, self.Mp, self.N)
        B = np.eye(self.N) - self.W @ self.A
        self.trW2 = np.trace(self.W @ self.W.T)
        self.trB2 = np.trace(B @ B.T)
        for p in range(self.P):
            self.oamps[p].receive_W(self.W_p[p])
        w_p = np.zeros((self.P, self.N, 1))
        for i in range(ite_max):
            for p in range(self.P):
                w_p[p], self.y_As_p[p] = self.oamps[p].local_compute()
            w_p[0] += self.s
            v = self._update_v()
            t = self._update_t(v, ord)
            self.s = self._update_s(C, w_p, t) if i != ite_max-1 else self._output_s(w_p, t)
            for p in range(self.P):
                self.oamps[p].receive_s(self.s)
            self.mse = self._add_mse()
            if ord == 'LMMSE':
                self.W = self.__set_W(v, ord='LMMSE')
                self.W_p = self.W.T.reshape(self.P, self.Mp, self.N)
                B = np.eye(self.N) - self.W @ self.A
                self.trW2 = np.trace(self.W @ self.W.T)
                self.trB2 = np.trace(B @ B.T)
                for p in range(self.P):
                    self.oamps[p].receive_W(self.W_p[p])

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

    def _update_v(self):
        y_As = np.sum(self.y_As_p)
        return (y_As - self.M * self.sigma) / self.trA2

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

    def _update_s(self, C, w, t):
        w_ = np.sum(w, axis=0)
        return C * DF(w_, t**0.5)

    def _output_s(self, w, t):
        return GCAMP(w, t**0.5)
