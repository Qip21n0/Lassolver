from lassolver.utils.func import *
from lassolver.utils.node import *
from lassolver.dsolver.d_base import *
import jax.numpy as jnp
import numpy as np



class AMPs(Node):
    def __init__(self, P, A_p, x, snr_p):
        super().__init__(A_p, x, snr_p)
        self.P = P
        self.normF_A = np.linalg.norm(A_p, ord='fro')**2
        self.Onsager = jnp.zeros(self.M)
        self.a = self.M / self.N
        self.v = [(np.linalg.norm(self.y)**2 - self.M * self.sigma) / self.normF_A]
        self.tau = []


    def local_compute(self):
        self._update_r_p()
        self._update_w_p()

        self._update_v_p()
        self._update_tau_p()


    def _update_r_p(self):
        self.r = self.y - self.A @ self.s


    def _update_w_p(self):
        self.s / self.P + self.AT @ (self.r + self.Onsager)


    def _update_v_p(self):
        v_p = (np.linalg.norm(self.r)**2 - self.M * self.sigma) / self.normF_A
        if v < 0:
            v = 1.e-4
        self.v.append(v_p)


    def _update_tau_p(self):
        tau_p = self.v[-1] / self.a + self.sigma_p
        self.tau.append(tau_p)


    def _update_s(self):
        pass


class DistributedAMP:
    def __init__(self, P, A, x, snr):
        self.P = P
        self.M, self.N = A.reshape
        self.M_p = self.M // self.P

        self.As = A.reshape(self.P, self.M_p, self.N)
        self.x = x.copy()
        self.snrs = self.__set_snrs(snr)
        self.nodes = [AMPs(self.P, self.As[p], self.x, self.snrs[p]) for p in range(P)]

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



class damp(dbase):
    def __init__(self, A_p, x, snr, M):
        super().__init__(A_p, x, snr, M)
        self.Onsager_p = np.zeros((self.M_p, 1))

    def receive_trA2(self, trA2):
        self.trA2 = trA2

    def update_Onsager(self, s):
        self.s = s.copy()
        self.Onsager_p = np.sum(self.s != 0) / self.M * (self.r_p + self.Onsager_p)

    def local_compute(self): 
        self.r_p = self._update_r_p()
        w_p = self._update_w_p()
        v_p = self._update_v_p()
        tau_p = self._update_tau_p(v_p)
        return w_p, v_p, tau_p

    def _update_r_p(self):
        return self.y - self.A_p @ self.s

    def _update_w_p(self):
        return self.s / self.P + self.AT_p @ (self.r_p + self.Onsager_p)

    def _update_v_p(self):
        v_p = (np.linalg.norm(self.r_p)**2 - self.M * self.sigma_p) / self.trA2
        return v_p

    def _update_tau_p(self, v_p):
        return self.N / self.M * v_p + self.sigma_p
        
        
class D_AMP(D_Base):
    def __init__(self, A, x, snr, P):
        super().__init__(A, x, snr, P)
        self.amps = [damp(self.A_p[p], x, snr, self.M) for p in range(self.P)]
        self.sigma = self.__set_sigma()
        self.trA2 = self.__set_trA2()

    def __set_sigma(self):
        sigma = 0
        for p in range(self.P):
            sigma += self.amps[p].sigma_p
        return sigma

    def __set_trA2(self):
        trA2 = 0
        for p in range(self.P):
            trA2 += self.amps[p].trA2_p
        return trA2

    def estimate(self, T=20, log=False):
        w = np.zeros((self.P, self.N, 1))

        for p in range(self.P):
            self.amps[p].receive_trA2(self.trA2)

        for t in range(T):
            for p in range(self.P):
                w[p], self.v[p], self.tau[p] = self.amps[p].local_compute()
            #w[0] += self.s
            v = self._update_v()
            tau = self._update_tau()
            if log: print("{}/{}: tau = {}, v = {}".format(t+1, T, tau, v))
            self._update_s(w, log)

            for p in range(self.P):
                self.amps[p].update_Onsager(self.s)
            self._add_mse()

    def _update_v(self):
        #r2 = np.sum(self.r2)
        #v = (r2 - self.M * self.sigma) / self.trA2
        v = np.sum(self.v)
        return v if v > 0 else 1e-4

    def _update_tau(self):
        #return v / self.a + self.sigma
        return np.sum(self.tau)

    def _update_s(self, w, log):
        s, communication_cost = GCAMP(w, self.tau, log)
        self.s = s
        self.communication_cost = np.append(self.communication_cost, communication_cost)