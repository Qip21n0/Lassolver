import numpy as np
import networkx as nx
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *


class doamp_sp(dbase):
    def __init__(self, A_p, x, noise, M):
        super().__init__(A_p, x, noise, M)
        self.a = self.M / self.N
        self.c = (self.N - self.M) / self.M
        self.omega_p = np.zeros((self.N, 1))
        self.gamma_p = 0
        self.theta_p = 0

    def receive_C(self, C):
        self.C = C

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
        return self.s / self.P + self.W_p @ self.r_p

    def _update_v_p(self):
        v_p = (np.linalg.norm(self.r_p)**2 - self.M_p * self.sigma_p) / self.trA2
        return v_p

    def _update_tau_p(self, v_p):
        return 1 / self.N * (self.trB2 * v_p + self.trW_p2 * self.sigma_p)
        #return np.linalg.norm(self.r_p + self.Onsager_p)**2 / self.M

    def _update_s_p(self):
        self.s = self.C * df(self.omega_p, self.theta_p**0.5)


class D_OAMP_SP(D_Base):
    def __init__(self, A, x, noise, Adj):
        P = len(Adj)
        super().__init__(A, x, noise, P)
        self.oamps = [doamp_sp(self.A_p[p], x, self.noise[p], self.M) for p in range(self.P)]
        self.Adj = Adj.copy()
        rows, cols = np.where(Adj == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        self.graph = gr
        self.A = A.copy()
        self.AT = self.A.T
        self.AAT = self.A @ self.AT
        self.I = np.eye(self.M)
        self.sigma = self.__set_sigma()
        self.trA2 = self.__set_trA2()
        self.mse = np.array([[None]*self.P])
        self.tau = np.array([[None]*self.P])
        self.v = np.array([[None]*self.P])

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

    def estimate(self, T=20, lT=10, C=1.85, ord='LMMSE', log=False):
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
            self.oamps[p].receive_C(C)
            self.oamps[p].receive_trA2(self.trA2)

        for t in range(T):
            w_pp = np.zeros((self.P, self.P, self.N, 1))
            v_pp = np.zeros((self.P, self.P))
            tau_pp = np.zeros((self.P, self.P))

            for p in range(self.P):
                w_pp[p, p], v_pp[p, p], tau_pp[p, p] = self.oamps[p].local_compute()
            for _ in range(lT):
                for p in range(self.P):
                    for j, v in enumerate(self.Adj[p]):
                        if v == 1:
                            w_pp[p][j] = np.sum(w_pp[:, p], axis=0) - w_pp[j][p]
                            v_pp[p][j] = np.sum(v_pp[:, p]) - v_pp[j][p]
                            tau_pp[p][j] = np.sum(tau_pp[:, p]) - tau_pp[j][p]
            v = self._update_v(v_pp)
            tau = self._update_tau(tau_pp)
            if log: 
                print(f"{t+1}/{T}")
                print(f"tau = {tau}")
                print(f"v = {v}")
                print("="*42)

            for p in range(self.P):
                self.oamps[p].omega_p = np.sum(w_pp[:, p], axis=0)
                gamma_p = np.sum(v_pp[:, p])
                self.oamps[p].gamma_p = gamma_p if gamma_p > 0 else 1e-4
                self.oamps[p].theta_p = np.sum(tau_pp[:, p])
                self.oamps[p]._update_s_p()
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

    def _update_v(self, v_pp):
        #r2 = np.sum(self.r2)
        #v = (r2 - self.M * self.sigma) / self.trA2
        v_p = np.zeros((1, self.P))
        for p in range(self.P):
            gamma_p = np.sum(v_pp[:, p])
            v_p[0, p] = gamma_p if gamma_p > 0 else 1e-4
        self.v = np.append(self.v, v_p, axis=0)
        return v_p

    def _update_tau(self, tau_pp):
        #return v / self.a + self.sigma
        tau_p = np.zeros((1, self.P))
        for p in range(self.P):
            tau_p[0, p] = np.sum(tau_pp[:, p])
        self.tau = np.append(self.tau, tau_p, axis=0)
        return tau_p

    def _add_mse(self):
        mse = np.zeros((1, self.P))
        for p in range(self.P):
            mse[0, p] = np.linalg.norm(self.oamps[p].s - self.x)**2 / self.N
        self.mse = np.append(self.mse, mse, axis=0)

    def result(self):
        for p in range(self.P):
            print(f"final mse(node {p}): {self.mse[-1, p]}")

        plt.figure(figsize=(16, 4))
        plt.subplot(121)
        plt.plot(self.x.real)
        for p in range(self.P):
            plt.plot(self.oamps[p].s.real)
        plt.grid()

        plt.subplot(122)
        plt.xlabel('iteration')
        plt.ylabel('MSE[log10]')
        ite = np.arange(0, np.shape(self.mse)[0], 1)
        plt.xticks(ite)
        for p in range(self.P):
            result = np.array([np.log10(mse) if mse is not None else None for mse in self.mse[:, p]])
            plt.plot(result)
            se = np.array([np.log10(v) if v is not None else None for v in self.v[:, p]])
            plt.scatter(ite, se)
        plt.grid()
    
    def show_graph(self):
        print(f"Diameter: {nx.diameter(self.graph)}")
        nx.draw(self.graph, node_size=200)
        plt.show()