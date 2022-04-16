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
        self.w = self.s / self.P + self.AT @ (self.r + self.Onsager)


    def _update_v_p(self):
        v_p = (np.linalg.norm(self.r)**2 - self.M * self.sigma) / self.fro_norm_A2
        if v_p < 0:
            v_p = 1.e-4
        self.v.append(v_p)


    def _update_tau_p(self):
        tau_p = self.v[-1] * self.a + self.sigma_p
        self.tau.append(tau_p)


    def update_Onsager_p(self):
        self.Onsager = jnp.sum(self.s != 0) / (self.M * self.P) * (self.r + self.Onsager)



class Core_damp(Edge_damp):
    def __init__(self, A, x, snr, fro_norm_A2, P, edges):
        super().__init__(A, x, snr, fro_norm_A2, P)
        self.netowork = edges.copy()


    def broadcast(self, s):
        for edge in self.netowork:
            edge.s = s.copy()
            edge.s_history.append(edge.s)



class DistributedAMP:
    def __init__(self, A, x, snr, P):
        self.P = P
        self.M, self.N = A.shape
        self.M_p = self.M // self.P
        fro_norm_A2 = np.linalg.norm(A, ord='fro')**2

        self.As = A.reshape(self.P, self.M_p, self.N)
        self.x = x.copy()
        self.snr = snr - 10 * np.log10(self.P)
        
        self.edges = [Edge_damp(self.As[p], self.x, self.snr, fro_norm_A2, self.P) for p in range(1, P)]
        self.core = Core_damp(self.As[0], self.x, self.snr, fro_norm_A2, self.P, self.edges)

        self.s = np.zeros(self.N)
        self.s_history = []

        norm_y2 = np.linalg.norm(self.core.y)**2 + np.sum([np.linalg.norm(self.edges[p].y)**2 for p in range(P-1)])
        self.sigma = (self.core.sigma + np.sum([self.edges[p].sigma for p in range(P-1)])) / P
        self.v = [(norm_y2 - self.M * self.sigma) / fro_norm_A2]
        self.tau = []
        self.mse = [None]
        self.comm_cost = []


    def estimate(self, T=20, CommCostCut=True, theta=0.7, log=False):
        self.theta = theta
        self.log = log

        for t in range(T):
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
                self._global_computation_amp()
            else:
                w = self.core.w
                w += jnp.sum([self.edges[p].w for p in range(self.P-1)], axis=0)
                if log:
                    print("Chose an option that does not reduce communication cost")
                    print(f"Total Communication Cost: {self.N * (self.P - 1)}")
                    print("="*50)
                
                self.s = soft_threshold(w, sum(self.tau[-1])**0.5)

            self._add_mse()
            self.s_history.append(self.s)
            self.core.broadcast(self.s)

            self.core.update_Onsager_p()
            for p in range(self.P-1):
                self.edges[p].update_Onsager_p()


    def _update_v(self):
        v = [self.core.v[-1]]
        for p in range(self.P-1):
            v.append(self.edges[p].v[-1])
        if sum(v) < 0:
            v = [1.e-4 / self.P] * self.P
        self.v.append(v)

    def _update_tau(self):
        tau = [self.core.tau[-1]]
        for p in range(self.P-1):
            tau.append(self.edges[p].tau[-1])
        self.tau.append(tau)


    def _global_computation_amp(self):
        # STEP1
        R = self._step1()

        # STEP2
        *_, U, F = self._step2(R)

        # STEP3
        self._step3(F, R)

        # STEP4
        self.s = self._step4(U)


    def _step1(self):
        tau = self.tau[-1]
        R = np.zeros((self.P, self.N))

        for p in range(self.P-1):
            R[p] = jnp.square(self.edges[p].w) > tau[p+1] * self.theta
            candidate = np.where(R[p])[0]
            for n in candidate:
                self.comm_cost += 1
        return R


    def _step2(self, R):
        tau = self.tau[-1]
        S = [np.where(R[:, n])[0] for n in range(self.N)]
        m = np.sum(R, axis=0)
        z = [0] * self.N
        U = np.empty(self.N)
        
        for n in range(self.N):
            upper = np.sum([tau[p] for p in range(1, self.P) if p not in S[p]])
            z[n] = self.edges[0].w[n] + np.sum([self.edges[p].w[n] for p in S[n]])
            U[n] = z[n]**2 + upper * self.theta

            F = (U > sum(tau)) * (m < (self.P-1))
            candidate = np.where(F)[0]
            for n in candidate:
                self.comm_cost += 1

        return z, S, U, F


    def _step3(self, F, R):
        F_not_in_R = F * np.logical_not(R)
        for p in range(1, self.P):
            candidate = np.where(F_not_in_R[p])[0]
            for n in candidate:
                self.comm_cost += 1
                
        if self.log: 
            print(f"R: {np.sum(R)} \t F: {np.sum(F)} \t F\\R: {np.sum(F_not_in_R)}")
            print(f"Total Communication Cost: {self.comm_cost}")

    
    def _step4(self, U):
        tau_sum = sum(self.tau[-1])
        s = np.zeros(self.N)
        count = 0

        V = np.where(U > tau_sum)[0].tolist()
        for n in V:
            w = self.core.w[n] + jnp.sum([self.edges[p].w[n] for p in range(self.P-1)], axis=0)
            s[n] = soft_threshold(w, tau_sum**0.5)
            count += s[n] == 0

        if self.log:
            print(f"the number of 0 obtained by soft threshold function: {count}/{np.sum(s != 0)}")
            print("="*50)

        return s.real


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