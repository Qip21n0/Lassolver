from lassolver.utils.node import *
import matplotlib.pyplot as plt
import numpy as np



class dbase:
    def __init__(self, A_p, x, snr, M):
        self.A_p = A_p.copy()
        self.M = M
        self.M_p, self.N = self.A_p.shape
        self.P = int(self.M / self.M_p)
        self.x = x
        Ax = self.A_p @ self.x
        SNRdB = 10**(0.1*snr)
        self.sigma_p = np.linalg.norm(Ax)**2 / SNRdB
        n = np.random.normal(0, self.sigma_p**0.5, (self.M_p, 1))
        self.y = Ax + n
        self.s = np.zeros((self.N, 1))
        self.AT_p = self.A_p.T
        self.trA2_p = np.trace(self.AT_p @ self.A_p)


class D_Base:
    def __init__(self, A, x, snr, P):
        self.M, self.N = A.shape
        self.P = P
        self.a = self.M / self.N
        self.M_p = int(self.M / self.P)
        self.A_p = A.reshape(P, self.M_p, self.N)
        self.x = x
        self.s = np.zeros((self.N, 1))
        self.mse = np.array([None])
        self.communication_cost = np.array([])
        self.r2 = np.zeros(self.P)
        self.tau = np.zeros(self.P)
        self.v = np.zeros(self.P)

    def _add_mse(self):
        mse = np.linalg.norm(self.s - self.x)**2 / self.N
        self.mse = np.append(self.mse, mse)

    def result(self):
        print("final mse: {}".format(self.mse[-1]))

        plt.figure(figsize=(16, 4))
        plt.subplot(121)
        plt.plot(self.x.real)
        plt.plot(self.s.real)
        plt.grid()

        plt.subplot(122)
        plt.xlabel('iteration')
        plt.ylabel('MSE[log10]')
        ite = np.shape(self.mse)[0]
        plt.xticks(np.arange(0, ite, 1))
        result = np.array([np.log10(mse) if mse is not None else None for mse in self.mse])
        plt.plot(result)
        plt.grid()
