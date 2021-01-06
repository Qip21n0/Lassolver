import numpy as np
import matplotlib.pyplot as plt

class base:
    def __init__(self, A, x, snr, M):
        self.A = A
        self.M = M
        self.Mp, self.N = A.shape
        self.x = x
        Ax = A @ x
        SNRdB = 10**(-0.1*snr)
        self.sigma = np.var(Ax) * SNRdB
        n = np.random.normal(0, self.sigma**0.5, (self.Mp, 1))
        self.y = Ax + n
        self.s = np.zeros((self.N, 1))
        self.AT = A.T
        self.trA2 = np.trace(self.AT @ self.A)


class D_Base:
    def __init__(self, A, x, snr, P):
        self.M, self.N = A.shape
        self.P = P
        self.a = self.M / self.N
        self.Mp = int(self.M / self.P)
        self.A_p = A.reshape(P, self.Mp, self.N)
        self.x = x
        self.s = np.zeros((self.N, 1))
        self.mse = np.array([np.linalg.norm(self.s - self.x)**2 / self.N])
        self.y_As_p = np.zeros(self.P)

    def _add_mse(self):
        mse = np.linalg.norm(self.s - self.x)**2 / self.N
        return np.append(self.mse, mse)

    def result(self):
        print("final mse: {}".format(self.mse[-1]))

        plt.figure(figsize=(16, 4))
        plt.subplot(121)
        plt.plot(self.x.real)
        plt.plot(self.s.real)
        plt.grid()

        plt.subplot(122)
        plt.xlabel('iteration')
        plt.ylabel('MSE')
        ite = np.shape(self.mse)[0]
        plt.xticks(np.arange(0, ite, 1))
        plt.plot(self.mse)
        plt.grid()
