import numpy as np
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *

class doamp(dbase):
    def __init__(self, A_p, x, snr, M):
        super().__init__(A_p, x, snr, M)

    def receive_s(self, s):
        self.s = s

    def receive_W_p(self, W_p):
        self.W_p = W_p.copy().T

    def local_compute(self):
        self.r_p = self._update_r_p()
        w_p = self._update_w_p()
        r2_p = np.linalg.norm(self.r_p)**2
        return w_p, r2_p

    def _update_r_p(self):
        return self.y - self.A_p @ self.s

    def _update_w_p(self):
        return self.W_p @ self.r_p


class D_OAMP(D_Base):
    def __init__(self, A, x, snr, P, iidG=False):
        super().__init__(A, x, snr, P)
        self.A = A.copy()
        self.AT = self.A.T
        self.AAT = self.A @ self.AT
        self.I = np.eye(self.M)
        self.c = (self.N - self.M) / self.M
        self.oamps = [doamp(self.A_p[p], x, snr, self.M) for p in range(self.P)]
        self.sigma = self.__set_sigma()
        self.trA2 = self.__set_trA2()
        self.iidG = iidG

    def __set_sigma(self):
        sigma = 0
        for p in range(self.P):
            sigma += self.oamps[p].sigma_p
        return sigma / self.P

    def __set_trA2(self):
        trA2 = 0
        for p in range(self.P):
            trA2 += self.oamps[p].trA2_p
        return trA2

    def estimate(self, T=20, C=2.0, ord='LMMSE', log=False, approx=False):
        lambda_ = [(i%2)*2 + 1 for i in range(1, T+1)]#np.linspace(3, 1, T)
        w = np.zeros((self.P, self.N, 1))
        self.communication_cost = np.empty(0)

        self.W = self.__set_W(1, ord)
        self.W_p = self.W.T.reshape(self.P, self.M_p, self.N)
        if not self.iidG:
            I = np.eye(self.N)
            B = I - self.W @ self.A
            self.trW2 = np.trace(self.W @ self.W.T)
            self.trB2 = np.trace(B @ B.T)
        for p in range(self.P):
            self.oamps[p].receive_W_p(self.W_p[p])
        
        for t in range(T):
            for p in range(self.P):
                w[p], self.r2[p] = self.oamps[p].local_compute()
            w[0] += self.s
            v = self._update_v()
            tau = self._update_tau(v, ord)
            if t == T-1:
                break
            self._update_s(C, w, lambda_[t], tau, log, approx)

            for p in range(self.P):
                self.oamps[p].receive_s(self.s)
            self._add_mse()

            if ord == 'LMMSE':
                self.W = self.__set_W(v, ord='LMMSE')
                self.W_p = self.W.T.reshape(self.P, self.M_p, self.N)
                if not self.iidG:
                    B = I - self.W @ self.A
                    self.trW2 = np.trace(self.W @ self.W.T)
                    self.trB2 = np.trace(B @ B.T)
                for p in range(self.P):
                    self.oamps[p].receive_W_p(self.W_p[p])
        
        self._output_s(w, lambda_[-1], tau, log)
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

    def _update_v(self):
        r2 = np.sum(self.r2)
        return (r2 - self.M * self.sigma) / self.trA2

    def _update_tau(self, v, ord):
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

    def _update_s(self, C, w, lambda_, tau, log, approx):
        s, communication_cost = GCOAMP(w, lambda_, tau**0.5, log, approx)
        self.s = C * s
        self.communication_cost = np.append(self.communication_cost, communication_cost)

    def _output_s(self, w, lambda_, tau, log):
        s, communication_cost = GCAMP(w, lambda_, tau**0.5, log)
        self.s = s
        self.communication_cost = np.append(self.communication_cost, communication_cost)