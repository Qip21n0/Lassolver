from lassolver.utils.func import *
from lassolver.utils.node import *
from lassolver.dsolver.damp import *
from scipy.stats import truncnorm
import jax.numpy as jnp
import numpy as np


class Edge_doamp(Edge_damp):
    def __init__(self, A, x, snr, fro_norm_A2, P):
        super().__init__(A, x, snr, fro_norm_A2, P)
        self.W = A.copy().T
        self.fro_norm_B2 = 0

    
    def local_computation(self):
        self._update_r_p()
        self._update_w_p()

        self._update_v_p()
        self._update_tau_p()


    def _update_w_p(self):
        self.w = self.s / self.P + self.W @ self.r

    
    def _update_tau_p(self):
        fro_norm_W2 = np.linalg.norm(self.W, 'fro')**2
        tau_p = (self.v[-1] * self.fro_norm_B2 + self.sigma_p * fro_norm_W2) / self.N
        self.tau.append(tau_p)


class Core_doamp(Edge_doamp):
    def __init__(self, A, x, snr, fro_norm_A2, P, edges):
        super().__init__(A, x, snr, fro_norm_A2, P)
        self.network = edges


    def send(self, Ws, fro_norm_B2):
        self.W = Ws[0].T.copy()
        self.fro_norm_B2 = fro_norm_B2

        for p in range(self.P-1):
            self.network[p].W = Ws[p+1].T.copy()
            self.network[p].fro_norm_B2 = fro_norm_B2


class DistributedOAMP(DistributedAMP):
    def __init__(self, A, x, snr, P):
        super().__init__(A, x, snr, P)
        fro_norm_A2 = np.linalg.norm(A, ord='fro')**2
        self.edges = [Edge_doamp(self.As[p], self.x, self.snr, fro_norm_A2, self.P) for p in range(1, P)]
        self.core = Core_doamp(self.As[0], self.x, self.snr, fro_norm_A2, self.P, self.edges)

        self.A = A.copy()
        self.AT = self.A.T
        self.A2 = self.A @ self.AT
        self.I = jnp.eye(self.M)

    
    def estimate(self, T=20, CommCostCut=True, theta=0.7, log=False, 
                        C=1.85, ord='PINV', last=True):
        self.theta = theta
        self.log = log
        I = jnp.eye(self.N)

        for t in range(T):
            if ord == "LMMSE" or t == 0:
                W = self.__set_W(ord)
                Ws = W.T.reshape(self.P, self.M_p, self.N)
                B = I - W @ self.A
                fro_norm_B2 = np.linalg.norm(B, ord='fro')**2
                self.core.send(Ws, fro_norm_B2)
            
            # Local Computation
            self.core.local_computation()
            for p in range(self.P-1):
                self.edges[p].local_computation()
            
            # Global Computation
            self._update_v()
            self._update_tau()
            if log:
                print(f"{t+1}/{T}: tau = {sum(self.tau[-1])}, v = {sum(self.v[-1])}")

            if CommCostCut:
                if last and t == T-1:
                    self._global_computation_amp()
                else:
                    self._global_computation_oamp(C)
            else:
                w = self.core.w
                w += jnp.sum([self.edges[p].w for p in range(self.P-1)], axis=0)
                if log:
                    print("Chose an option that does not reduce communication cost")
                    print(f"Total Communication Cost: {self.N * (self.P - 1)}")
                    print("="*50)
                
                if last and t == T-1:
                    self.s = soft_threshold(w, sum(self.tau[-1])**0.5)
                else:
                    self.s = C * df(w, sum(self.tau[-1])**0.5)

            self._add_mse()
            self.s_history.append(self.s)
            self.core.broadcast(self.s)


    def __set_W(self, ord):
        v = sum(self.v[-1])
        
        if ord == 'MF':
            W_ = self.AT

        elif ord == 'PINV':
            W_ = jnp.linalg.pinv(self.A)

        elif ord == 'LMMSE':
            W_ = v * self.AT @ jnp.linalg.inv(v * self.A2 + self.sigma * self.I)

        else :
            raise NameError("not correct")

        return self.N / np.trace(W_ @ self.A) * W_


    def _global_computation_oamp(self, C):
        # STEP1
        R = self._step1()

        # STEP2
        z, S, U, F = self._step2(R)

        # STEP3
        self._step3(F, R)

        # STEP4-5
        self.s = C * self._step45( z, S, U)

    
    def _step45(self, z, S, U):
        tau = self.tau[-1]
        tau_sum = sum(tau)
        u = np.zeros(self.N)
        b = np.zeros(self.N)
        count = 0

        V = np.where(U > tau_sum)[0].tolist()
        for n in V:
            w = self.core.w[n] + jnp.sum([self.edges[p].w[n] for p in range(self.P-1)], axis=0)
            u[n] = soft_threshold(w, tau_sum**0.5)
            count += u[n] == 0

        if self.log:
            print(f"the number of 0 obtained by soft threshold function: {count}/{np.sum(s != 0)}")
            print("="*50)

        Vc = [n for n in range(self.N) if n not in V]
        for n in Vc:
            b[n] = z[n]
            b[n] += np.sum([self.rand(self.theta * tau[p]) for p in range(1, self.P) if p not in S[n]])
            s = u - np.mean(u != 0)*b
            
        return s.real
    
    def rand(self, num):
        return num**0.5 * truncnorm.rvs(-1, 1, loc=0, scale=1, size=1)