import numpy as np
import networkx as nx
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *


class damp_sp(dbase):
    def __init__(self, A_p, x, noise, M):
        super().__init__(A_p, x, noise, M)
        self.a = self.M / self.N
        self.Onsager_p = np.zeros((self.M_p, 1))
        self.omega_p = np.zeros((self.N, 1))
        self.gamma_p = 0
        self.theta_p = 0

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
        return self.s / self.P + self.AT_p @ (self.r_p + self.Onsager_p)

    def _update_v_p(self):
        v_p = (np.linalg.norm(self.r_p)**2 - self.M_p * self.sigma_p) / self.trA2
        return v_p

    def _update_tau_p(self, v_p):
        return v_p / self.a + self.sigma_p
        #return np.linalg.norm(self.r_p + self.Onsager_p)**2 / self.M

    def _update_s_p(self):
        self.s = soft_threshold(self.omega_p, self.theta_p**0.5)
        self.Onsager_p = np.sum(self.s != 0) / self.M * (self.r_p + self.Onsager_p)


class D_AMP_SP(D_Base):
    def __init__(self, A, x, noise, Adj):
        P = len(Adj)
        super().__init__(A, x, noise, P)
        self.amps = [damp_sp(self.A_p[p], x, self.noise[p], self.M) for p in range(self.P)]
        self.Adj = Adj.copy()
        rows, cols = np.where(Adj == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        self.graph = gr
        self.sigma = self.__set_sigma()
        self.trA2 = self.__set_trA2()
        self.mse = np.array([[None]*self.P])
        self.tau = [[None]*self.P]
        self.v = [[None]*self.P]

    def __set_sigma(self):
        sigma = 0
        for p in range(self.P):
            sigma += self.amps[p].sigma_p
        return sigma / self.P

    def __set_trA2(self):
        trA2 = 0
        for p in range(self.P):
            trA2 += self.amps[p].trA2_p
        return trA2

    def estimate(self, T=20, lT=10, log=False):
        for p in range(self.P):
            self.amps[p].receive_trA2(self.trA2)

        for t in range(T):
            w_pp = np.zeros((self.P, self.P, self.N, 1))
            v_pp = np.zeros((self.P, self.P))
            tau_pp = np.zeros((self.P, self.P))

            for p in range(self.P):
                w_pp[p, p], v_pp[p, p], tau_pp[p, p] = self.amps[p].local_compute()
            for _ in range(lT):
                for p in range(self.P):
                    for j, v in enumerate(self.Adj[p]):
                        if v == 1:
                            w_pp[p][j] = np.sum(w_pp[:, p], axis=0) - w_pp[j][p]
                            v_pp[p][j] = np.sum(v_pp[:, p]) - v_pp[j][p]
                            tau_pp[p][j] = np.sum(tau_pp[:, p]) - tau_pp[j][p]
            #v = self._update_v(v_pp)
            #tau = self._update_tau(tau_pp)
            #if log: print("{}/{}: tau = {}, v = {}".format(t+1, T, tau, v))

            for p in range(self.P):
                self.amps[p].omega_p = np.sum(w_pp[:, p], axis=0)
                gamma_p = np.sum(v_pp[:, p])
                self.amps[p].gamma_p = gamma_p if gamma_p > 0 else 1e-4
                self.amps[p].theta_p = np.sum(tau_pp[:, p])
                self.amps[p]._update_s_p()
            self._add_mse()

    def _update_v(self, v_pp):
        #r2 = np.sum(self.r2)
        #v = (r2 - self.M * self.sigma) / self.trA2
        for p in range(self.P):
            self.v_p[p] = v_pp[p, p]
        v = np.sum(self.v_p)
        v = v if v > 0 else 1e-4
        self.v.append(v)
        return v

    def _update_tau(self, tau_pp):
        #return v / self.a + self.sigma
        return (np.sum(self.tau_p) / self.M)**0.5

    def _add_mse(self):
        mse = np.zeros((1, self.P))
        for p in range(self.P):
            mse[0, p] = np.linalg.norm(self.amps[p].s - self.x)**2 / self.N
        self.mse = np.append(self.mse, mse, axis=0)

    def result(self):
        for p in range(self.P):
            print("final mse(node {p}): {}".format(self.mse[-1, p]))

        plt.figure(figsize=(16, 4))
        plt.subplot(121)
        plt.plot(self.x.real)
        for p in range(self.P):
            plt.plot(self.amps[p].s.real)
        plt.grid()

        plt.subplot(122)
        plt.xlabel('iteration')
        plt.ylabel('MSE[log10]')
        ite = np.arange(0, np.shape(self.mse)[0], 1)
        plt.xticks(ite)
        for p in range(self.P):
            result = np.array([np.log10(mse) if mse is not None else None for mse in self.mse[:, p]])
            plt.plot(result)
        #se = np.array([np.log10(v) if v is not None else None for v in self.v])
        #plt.scatter(ite, se, c='red')
        plt.grid()
    
    def show_graph(self):
        print(f"Diameter: {nx.diameter(self.graph)}")
        nx.draw(self.graph, node_size=500)
        plt.show()