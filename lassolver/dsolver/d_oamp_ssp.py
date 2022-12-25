import numpy as np
import networkx as nx
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *


class doamp_ssp(dbase):
    def __init__(self, A_p, x, noise, M):
        super().__init__(A_p, x, noise, M)
        self.a = self.M / self.N
        self.c = (self.N - self.M) / self.M
        self.omega_p = np.zeros((self.N, 1))
        self.gamma_p = 0
        self.theta_p = 0
        self.communication_cost_p = np.array([])
        self.estimated_positions = []

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


class D_OAMP_SSP(D_Base):
    def __init__(self, A, x, noise, Adj):
        P = len(Adj)
        super().__init__(A, x, noise, P)
        self.oamps = [doamp_ssp(self.A_p[p], x, self.noise[p], self.M) for p in range(self.P)]
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

    def estimate(self, T=20, lT=10, C=1.85, theta=0.7, ord='LMMSE', rand=True, log=False):
        order = np.arange(self.P)

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
            communication_cost = [0] * self.P

            for p in range(self.P):
                w_pp[p, p], v_pp[p, p], tau_pp[p, p] = self.oamps[p].local_compute()
            for _ in range(lT):
                if rand:
                    np.random.shuffle(order)
                for p in order:
                    w_pp[p], v_pp[p], tau_pp[p], comm_cost = self.selective_summation_propagation(p, w_pp[:, p], v_pp[:, p], tau_pp[:, p], theta)
                    communication_cost[p] += comm_cost
                    #for j, v in enumerate(self.Adj[p]):
                    #    if v == 1:
                    #        w_pp[p][j], comm_cost = self.selective_summation_propagation(p, j, w_pp[:, p], tau_pp[:, p], theta)
                    #        communication_cost[p] += comm_cost
                    #        v_pp[p][j] = np.sum(v_pp[:, p]) - v_pp[j, p]
                    #        tau_pp[p][j] = np.sum(tau_pp[:, p]) - tau_pp[j, p]
            for p in range(self.P):
                self.oamps[p].communication_cost_p = np.append(self.oamps[p].communication_cost_p, communication_cost[p])
            v = self._update_v(v_pp)
            tau = self._update_tau(tau_pp)
            if log: 
                print(f"{t+1}/{T}")
                print(f"tau = {np.sum(tau_pp[:, 0])}")
                print(f"v = {np.sum(v_pp[:, 0])}")
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

    def selective_summation_propagation(self, p, phi_p, nu_p, zeta_p, theta):
        """
        p, j: int
            Number of node

        phi_p: np.ndarray(P, N, 1)
            vector for node i to p (i in N_p)
        
        nu_p: np.ndaaray(P)
            estimated MSE for node i to p (i in N_p)
        
        zeta_p: np.adarray(P)
            threshold for node i to p (i in N_p)
        
        theta: float
            tuning parameter in (0, 1)
        """
        phi = np.sum(phi_p, axis=0)
        nu = np.sum(nu_p)
        zeta = np.sum(zeta_p)
        new_nu_p = nu * self.Adj[p] - nu_p
        new_nu_p[p] = nu_p[p]
        new_zeta_p = zeta * self.Adj[p] - zeta_p
        new_zeta_p[p] = zeta_p[p]
        N_p = np.where(self.Adj[p])[0]
        #N_p = np.delete(_, np.where(_ == p)[0])
        communication_cost = 0
        R = np.zeros((self.P, self.N, 1))
        z = np.empty(self.N)
        # STEP1
        for i in N_p:
            R[i] = np.square(phi_p[i]) > zeta_p[i] * theta
            candidate = np.where(R[i])[0]
            for n in candidate:
                communication_cost += 1
        # STEP2
        S = [np.where(R[:, n])[0] for n in range(self.N)]
        m = np.sum(R, axis=0)
        U = np.empty((self.N, 1))
        for n in range(self.N):
            upper = np.sum([zeta_p[i] for i in N_p if i not in S[i]])
            z[n] = phi_p[p, n] + np.sum([phi_p[i, n] for i in S[n]])
            U[n] = z[n]**2 + upper * theta
        F = U > zeta#(U > zeta) & (m < (len(N_p)))
        candidate = np.where(F)[0]
        for n in candidate:
            communication_cost += 1
        # STEP3
        F_R = F * np.logical_not(R)
        for i in N_p:
            candidate = np.where(F_R[i])[0]
            for n in candidate:
                communication_cost += 1
        # STEP4
        new_phi_p = np.zeros((self.P, self.N, 1))
        V = np.where(U > zeta)[0].tolist()
        for n in V:
            #for j in N_p:
            #    new_phi_p[j, n] = phi[n] - phi_p[j, n]
            new_phi_p[:, n] = (phi[n] * self.Adj[p]).reshape((self.P, 1)) - phi_p[:, n]
        Vc = [n for n in range(self.N) if n not in V]
        for n in Vc:
            for j in N_p:
                new_phi_p[j, n] = phi_p[p, n] + np.sum([phi_p[i, n] for i in S[n] if i != j])
        new_phi_p[p] = phi_p[p]
        #self.oamps[p].estimated_positions.append(V)
        return new_phi_p.real, new_nu_p, new_zeta_p, communication_cost

    def _add_mse(self):
        mse = np.zeros((1, self.P))
        for p in range(self.P):
            mse[0, p] = np.linalg.norm(self.oamps[p].s - self.x)**2 / self.N
        self.mse = np.append(self.mse, mse, axis=0)

    def result(self):
        last_mse = self.mse[-1, :].copy()
        print(f'final mse mean: {np.mean(last_mse)}, max: {np.max(last_mse)}, min: {np.min(last_mse)}')
        plt.hist(last_mse)

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