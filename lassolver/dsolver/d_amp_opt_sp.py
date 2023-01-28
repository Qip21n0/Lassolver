import jax
import numpy as np
from jax.scipy.stats import norm as normal
import networkx as nx
from lassolver.utils.func import *
from lassolver.dsolver.d_base import *


class damp_opt_sp(dbase):
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
        #return v_p / self.a + self.sigma_p / self.P
        return np.linalg.norm(self.r_p + self.Onsager_p)**2 / self.M

    def _update_s_p(self):
        rho = np.mean(soft_threshold(self.omega_p, self.theta_p**0.5) != 0)
        def func_mmse(vector, threshold):
            xi = rho**(-1) + threshold
            top = normal.pdf(vector, loc=0, scale=xi**0.5) /xi
            bottom = rho * normal.pdf(vector, loc=0, scale=xi**0.5) + (1-rho) * normal.pdf(vector, loc=0, scale=threshold**0.5)
            return top / bottom * vector
        self.s = func_mmse(self.omega_p, self.theta_p)
        
        dfunc_mmse = jax.vmap(jax.grad(func_mmse, argnums=(0)), (0, None))
        self.Onsager_p = np.sum(dfunc_mmse(self.omega_p.reshape(self.N), self.theta_p)) / self.M * (self.r_p + self.Onsager_p)


class D_AMP_OPT_SP(D_Base):
    def __init__(self, A, x, noise, Adj):
        P = len(Adj)
        super().__init__(A, x, noise, P)
        self.amps = [damp_opt_sp(self.A_p[p], x, self.noise[p], self.M) for p in range(self.P)]
        self.Adj = Adj.copy()
        rows, cols = np.where(Adj == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        self.graph = gr
        self.sigma = self.__set_sigma()
        self.trA2 = self.__set_trA2()
        self.mse = np.array([[None]*self.P])
        self.tau = np.array([[None]*self.P])
        self.v = np.array([[None]*self.P])

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

    def estimate(self, T=20, lT=10, rand=True, log=False):
        order = np.arange(self.P)
        for p in range(self.P):
            self.amps[p].receive_trA2(self.trA2)

        for t in range(T):
            w_pp = np.zeros((self.P, self.P, self.N, 1))
            v_pp = np.zeros((self.P, self.P))
            tau_pp = np.zeros((self.P, self.P))

            for p in range(self.P):
                w_pp[p, p], v_pp[p, p], tau_pp[p, p] = self.amps[p].local_compute()
            for _ in range(lT):
                if rand:
                    np.random.shuffle(order)
                for p in order:
                    w_pp[p], v_pp[p], tau_pp[p] = self.summation_propagation(p, w_pp[:, p], v_pp[:, p], tau_pp[:, p])
                    #for j, islinekd in enumerate(self.Adj[p]):
                    #    if islinekd == 1:
                    #        w_pp[p][j] = np.sum(w_pp[:, p], axis=0) - w_pp[j][p]
                    #        v_pp[p][j] = np.sum(v_pp[:, p]) - v_pp[j][p]
                    #        tau_pp[p][j] = np.sum(tau_pp[:, p]) - tau_pp[j][p]
            v = self._update_v(v_pp)
            tau = self._update_tau(tau_pp)
            if log: 
                print(f"{t+1}/{T}")
                print(f"tau = {np.sum(tau_pp[:, 0])}")
                print(f"v = {np.sum(v_pp[:, 0])}")
                print("="*42)

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

    def summation_propagation(self, p, phi_p, nu_p, zeta_p):
        phi = np.sum(phi_p, axis=0)
        nu = np.sum(nu_p)
        zeta = np.sum(zeta_p)
        N_p = np.where(self.Adj[p])[0]

        new_phi_p = np.zeros((self.P, self.N, 1))
        new_nu_p = np.zeros(self.P)
        new_zeta_p = np.zeros(self.P)
        for i in N_p:
            new_phi_p[i] = phi - phi_p[i]
            new_nu_p[i] = nu - nu_p[i]
            new_zeta_p[i] = zeta - zeta_p[i]
        #new_phi_p = (phi * self.Adj[p]).T.reshape((self.P, self.N, 1)) - phi_p
        new_phi_p[p] = phi_p[p]
        #new_nu_p = nu * self.Adj[p] - nu_p
        new_nu_p[p] = nu_p[p]
        #new_zeta_p = zeta * self.Adj[p] - zeta_p
        new_zeta_p[p] = zeta_p[p]

        return new_phi_p, new_nu_p, new_zeta_p

    def _add_mse(self):
        mse = np.zeros((1, self.P))
        for p in range(self.P):
            mse[0, p] = np.linalg.norm(self.amps[p].s - self.x)**2 / self.N
        self.mse = np.append(self.mse, mse, axis=0)

    def result(self):
        last_mse = self.mse[-1, :].copy()
        print(f'final mse mean: {np.mean(last_mse)}, max: {np.max(last_mse)}, min: {np.min(last_mse)}')
        plt.hist(last_mse)

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
            se = np.array([np.log10(v) if v is not None else None for v in self.v[:, p]])
            plt.scatter(ite, se)
        plt.grid()
    
    def show_graph(self):
        print(f"Diameter: {nx.diameter(self.graph)}")
        nx.draw(self.graph, node_size=200)
        plt.show()