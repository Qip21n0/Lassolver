from lassolver.utils.func import *
from lassolver.solver.amp import AMP
import numpy as np



class OAMP(AMP):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)
        self.W = self.__set_W(ord)
        self.A2 = A @ A.T
        self.I = np.eye(self.M)
        self.c = (self.N - self.M) / self.M
        
        self.normF_W = np.linalg.norm(self.W, ord='fro')**2


    def estimate(self, T=20, C=1.85, ord='LMMSE', out='st'):
        I = np.eye(self.N)
        B = I - self.W @ self.A
        self.normF_B = np.linalg.norm(self.B, ord='fro')**2
        
        for t in range(T):
            self._update_r()
            self._update_w()
            
            self._update_v()
            self._update_tau()
            if t == T-1: break
            
            self._update_s(C)
            self._add_mse()
            self.s_history.append(self.s)

            if ord == 'LMMSE':
                self.W = self.__set_W(ord='LMMSE')
                B = I - self.W @ self.A
                self.trW2 = np.trace(self.W @ self.W.T)
                self.trB2 = np.trace(B @ B.T)
        
        self.s = self._output_s(C, out)
        self._add_mse()
        self.s_history.append(self.s)


    def __set_W(self, ord):
        if ord == 'MF':
            W_ = self.AT
        elif ord == 'PINV':
            W_ = np.linalg.pinv(self.A)
        elif ord == 'LMMSE':
            W_ = self.v[-1] * self.AT @ np.linalg.inv(self.v[-1] * self.A2 + self.sigma * self.I)
        else :
            raise NameError("not correct")
        return self.N / np.trace(W_ @ self.A) * W_


    def _update_w(self):
        self.w = self.s + self.W @ self.r


    def _update_tau(self):
        tau = 1/self.N * (self.normF_B * self.v[-1] + self.normF_W * self.sigma)
        self.tau.append(tau)


    def _update_s(self, C):
        self.s = C * df(self.w, self.tau[-1]**0.5)


    def _output_s(self, C, out):
        if out == 'st':
            self.s = soft_threshold(self.w, self.tau[-1]**0.5)
        elif out == 'df':
            self.s = C * df(self.w, self.tau[-1]**0.5)
        else:
            NameError('not correct')
