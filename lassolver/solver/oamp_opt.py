import jax
import numpy as np
from jax.scipy.stats import norm as normal
from lassolver.utils.func import *
from lassolver.solver.amp import AMP


class OAMP_OPT(AMP):
    def __init__(self, A, x, snr):
        super().__init__(A, x, snr)
        self.AAT = A @ A.T
        self.I = np.eye(self.M)
        self.c = (self.N - self.M) / self.M

    def estimate(self, T=20, ord='LMMSE', log=False):
        v = self._update_v(self.y)
        self.v.pop()
        self.W = self.__set_W(v, ord)
        
        I = np.eye(self.N)
        B = I - self.W @ self.A
        self.trW2 = np.trace(self.W @ self.W.T)
        self.trB2 = np.trace(B @ B.T)
        
        for t in range(T):
            r = self._update_r()
            w = self._update_w(r)
            v = self._update_v(r)
            tau = self._update_tau(v)
            self.s, message = self._update_s(w, tau)
            if log: 
                print(f"{t+1}/{T}")
                print(f"tau = {tau}")
                print(f"v = {v}")
                print(message)
                print("="*42)
            self._add_mse()
            if t == T-1: break
            if ord == 'LMMSE':
                self.W = self.__set_W(v, ord='LMMSE')
                B = np.eye(self.N) - self.W @ self.A
                self.trW2 = np.trace(self.W @ self.W.T)
                self.trB2 = np.trace(B @ B.T)
        
        #self.s = self._output_s(w, tau)
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

    def _update_w(self, r):
        return self.s + self.W @ r

    def _update_tau(self, v):
        tau = 1/self.N * (self.trB2 * v + self.trW2 * self.sigma)
        self.tau.append(tau)
        return tau

    def _update_s(self, w, tau):
        rho = np.mean(soft_threshold(w, tau**0.5) != 0)
        def func_mmse(vector, threshold):
            xi = rho**(-1) + threshold
            top = normal.pdf(vector, loc=0, scale=xi**0.5) / xi
            bottom = rho * normal.pdf(vector, loc=0, scale=xi**0.5) + (1-rho) * normal.pdf(vector, loc=0, scale=threshold**0.5)
            return top / bottom * vector

        dfunc_mmse = jax.vmap(jax.grad(func_mmse, argnums=(0)), (0, None))
        reshaped_w = w.reshape(self.N)
        v_mmse = tau**0.5 * np.mean(dfunc_mmse(reshaped_w, tau))
        C_mmse = tau**0.5 / (tau**0.5 - v_mmse)
        message = f"DF_MMSE(w) = {C_mmse} * (f_MMSE(w) - {np.mean(dfunc_mmse(reshaped_w, tau))} * w)   rho = {rho}"
        return C_mmse * (func_mmse(w, tau) - np.mean(dfunc_mmse(reshaped_w, tau)) * w), message

    def _output_s(self, w, tau):
        return soft_threshold(w, tau**0.5)
