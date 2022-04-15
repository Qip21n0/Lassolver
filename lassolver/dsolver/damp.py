from lassolver.utils.func import *
from lassolver.utils.node import *
import jax.numpy as jnp
import numpy as np



class Edge_damp(Node):
    def __init__(self, A, x, snr, fro_norm_A2, P):
        super().__init__(A, x, snr)
        self.r = jnp.zeros(self.M)
        self.w = jnp.zeros(self.N)
        self.s = jnp.zeros(self.N, dtype=jnp.float32)
        self.s_history = []

        self.AT = self.A.T

        self.Onsager = jnp.zeros(self.M)
        self.P = P
        self.a = self.N / self.M / self.P
        self.fro_norm_A2 = fro_norm_A2

        self.v = [(np.linalg.norm(self.y)**2 - self.M * self.sigma) / self.fro_norm_A2]
        self.tau = []


    def local_computation(self):
        self._update_r_p()
        self._update_w_p()

        self._update_v_p()
        self._update_tau_p()


    def _update_r_p(self):
        self.r = self.y - self.A @ self.s


    def _update_w_p(self):
        self.s / self.P + self.AT @ (self.r + self.Onsager)


    def _update_v_p(self):
        v_p = (np.linalg.norm(self.r)**2 - self.M * self.sigma) / self.fro_norm_A2
        if v_p < 0:
            v_p = 1.e-4
        self.v.append(v_p)


    def _update_tau_p(self):
        tau_p = self.v[-1] * self.a + self.sigma_p
        self.tau.append(tau_p)


    def _update_s(self):
        pass


    def _update_Onsager_p(self, s):
        pass


    def send(self, value):
        pass



class Core_damp(Edge_damp):
    def __init__(self, A, x, snr, fro_norm_A2, P, edges):
        super().__init__(A, x, snr, fro_norm_A2, P)
        self.netowork = edges.copy()


    def broadcast(self):
        for edge in self.netowork:
            edge.tau = self.tau



class DistributedAMP:
    def __init__(self, A, x, snr, P):
        self.P = P
        self.M, self.N = A.reshape
        self.M_p = self.M // self.P
        fro_norm_A2 = np.linalg.norm(A, ord='fro')**2

        self.As = A.reshape(self.P, self.M_p, self.N)
        self.x = x.copy()
        self.snrs = self.__set_snrs(snr)
        
        self.edges = [Edge_damp(self.As[p], self.x, self.snrs[p], fro_norm_A2, self.P) for p in range(1, P)]
        self.core = Core_damp(self.As[0], self.x, self.snrs[0], fro_norm_A2, self.P, self.edges)

        self.s = jnp.zeros(self.N, dtype=jnp.float32)
        self.s_history = []

        self.mse = [None]
        self.comm_cost = []


    def __set_snrs(self, snr):
        if type(snr) is int or float:
            snrs = [snr] * self.P

        elif len(snr) > 0 and type(snr) is not str:
            div_num = self.P // len(snr)
            snrs = []
            for i in range(len(snr)):
                for _ in range(div_num):
                    snrs.append(snr[i])
            if len(snrs) != self.P:
                snrs += [snr[-1]] * (self.P - len(snrs))

        else:
            ValueError
        
        return snrs


    def estimate(self, T=20, CommCostCut=True, log=False):
        for t in range(T):
            # Local Computation
            self.core.local_computation()
            for p in range(self.P-1):
                self.edges[p].local_computation()
            
            # Global Computation
            tau = [self.core.tau]
            for p in range(self.P-1):
                tau.append(self.edges[p].tau)
            v = [self.core.v]
            for p in range(self.P-1):
                v.append(self.edges[p].v)

            if CommCostCut:
                self.GCAMP()
            
            else:
                w = self.core.w
                w += jnp.sum([self.edges[p].w for p in range(self.P-1)])
                s = soft_threshold(w, sum(tau)**0.5)


    def GCAMP(self, w, tau):
        pass


    def _add_mse(self):
        mse = np.linalg.norm(self.s - self.x)**2 / self.N
        self.mse.append(mse)


    def show_result(self):
        print("final mse: {}".format(self.mse[-1]))
        plt.figure(figsize=(16, 4))

        plt.subplot(121)
        plt.plot(self.x.real)
        plt.plot(self.s.real)
        plt.grid()

        plt.subplot(122)
        plt.xlabel('iteration')
        plt.ylabel('MSE[log10]')
        iter = np.shape(self.mse)[0]
        plt.xticks(np.arange(0, iter, 1))
        result = np.array([np.log10(mse) if mse is not None else None for mse in self.mse])
        plt.plot(result)
        plt.grid()