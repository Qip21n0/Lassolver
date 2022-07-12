import numpy as np
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *

class doamp_exp(dbase):
    def __init__(self, A_p, x, snr, M):
        super().__init__(A_p, x, snr, M)

    def receive_s(self, s):
        self.s = s.copy()

    def receive_W_p(self, W_p):
        self.W_p = W_p.copy()

    def receive_trX2(self, trW_p2, trB2):
        self.trW_p2 = trW_p2
        self.trB2 = trB2

    def receive_trA2(self, trA2):
        self.trA2 = trA2

    def local_compute(self):
        self.r_p = self._update_r_p()
        w_p = self._update_w_p()
        v_p = self._update_v_p()
        tau_p = self._update_tau_p(v_p)
        return w_p, v_p, tau_p

    def _update_r_p(self):
        return self.y - self.A_p @ self.s

    def _update_w_p(self):
        return self.W_p @ self.r_p

    def _update_v_p(self):
        v_p = (np.linalg.norm(self.r_p)**2 - self.M_p * self.sigma_p) / self.trA2
        return v_p

    def _update_tau_p(self, v_p):
        return 1 / self.N * (self.trB2 * v_p + self.trW_p2 * self.sigma_p)


class D_OAMP_exp(D_Base):
    def __init__(self, A, x, noise, P):
        super().__init__(A, x, noise, P)
        self.A = A.copy()
        self.AT = self.A.T
        self.AAT = self.A @ self.AT
        self.I = np.eye(self.M)
        self.c = (self.N - self.M) / self.M
        self.oamps = [doamp_exp(self.A_p[p], x, self.noise[p], self.M) for p in range(self.P)]
        self.sigma = self.__set_sigma()
        self.trA2 = self.__set_trA2()

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

    def estimate(self, T=20, C=1.85, ord='LMMSE', log=False):
        w = np.zeros((self.P, self.N, 1))

        v = (np.sum([np.linalg.norm(self.oamps[p].y)**2 for p in range(self.P)]) - self.M_p * self.sigma) / self.trA2
        self.W = self.__set_W(v, ord)
        self.W_p = self.W.T.reshape(self.P, self.M_p, self.N)
        
        I = np.eye(self.N)
        B = I - self.W @ self.A
        self.trW2 = [np.trace(W_p.T @ W_p) for W_p in self.W_p]
        self.trB2 = np.trace(B @ B.T)

        for p in range(self.P):
            self.oamps[p].receive_W_p(self.W_p[p].T)
            self.oamps[p].receive_trX2(self.trW2[p], self.trB2)
            self.oamps[p].receive_trA2(self.trA2)
        
        for t in range(T):
            for p in range(self.P):
                w[p], self.v_p[p], self.tau_p[p] = self.oamps[p].local_compute()
            w[0] += self.s
            v = self._update_v()
            tau = self._update_tau()
            if log: print("{}/{}: tau = {}, v = {}".format(t+1, T, tau, v))
            self._update_s(C, w, log)

            for p in range(self.P):
                self.oamps[p].receive_s(self.s)
            self._add_mse()
            if t == T-1: break
            if ord == 'LMMSE':
                self.W = self.__set_W(v, ord='LMMSE')
                self.W_p = self.W.T.reshape(self.P, self.M_p, self.N)
                B = I - self.W @ self.A
                self.trW2 = [np.trace(W_p.T @ W_p) for W_p in self.W_p]
                self.trB2 = np.trace(B @ B.T)
                for p in range(self.P):
                    self.oamps[p].receive_W_p(self.W_p[p].T)
                    self.oamps[p].receive_trX2(self.trW2[p], self.trB2)
        
        #self._output_s(w, log)
        #self._add_mse()

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
        #r2 = np.sum(self.r2)
        #v = (r2 - self.M * self.sigma) / self.trA2
        v = np.sum(self.v_p)
        v = v if v > 0 else 1e-4
        self.v.append(v)
        return v

    def _update_tau(self):
        #return 1/self.N * (self.trB2 * v + self.trW2 * self.sigma)
        return np.sum(self.tau_p)

    def _update_s(self, C, w, log):
        s, communication_cost, b_ws = GCOAMP(w, self.tau_p, log)
        self.s = C * s
        self.communication_cost = np.append(self.communication_cost, communication_cost)

        size = 20
        hist, bins = np.histogram(b_ws, bins=size)
        index_4_hist = np.digitize(b_ws, bins) - 1
        mse_4_hist = np.zeros(size)
        for i in range(self.N):
            j = index_4_hist[i] - 1
            mse_4_hist[j] += self._square_error_4_component(i)

        for i in range(size):
            mse_4_hist[i] /= hist[i]

        self.mse_4_hist.append(mse_4_hist)

    def _output_s(self, w, log):
        s, communication_cost = GCAMP(w, self.tau_p, log)
        self.s = s
        self.communication_cost = np.append(self.communication_cost, communication_cost)