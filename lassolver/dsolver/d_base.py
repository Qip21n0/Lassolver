import numpy as np
import matplotlib.pyplot as plt

class dbase:
    def __init__(self, A_p, x, noise, M):
        self.A_p = A_p.copy()
        self.M = M
        self.M_p, self.N = self.A_p.shape
        self.P = int(self.M / self.M_p)
        self.x = x
        Ax = self.A_p @ self.x
        if type(noise) is int:
            SNRdB = 10**(0.1*noise) / self.P
            self.sigma_p = np.linalg.norm(Ax)**2 / SNRdB
            n = np.random.normal(0, self.sigma_p**0.5, (self.M_p, 1))
        elif type(noise).__module__ == 'numpy':
            self.sigma_p = np.var(noise)
            n = noise.copy()
        else :
            raise ValueError
        self.y = Ax + n
        self.s = np.zeros((self.N, 1))
        self.AT_p = self.A_p.T
        self.trA2_p = np.trace(self.AT_p @ self.A_p)


class D_Base:
    def __init__(self, A, x, noise, P):
        self.M, self.N = A.shape
        self.K = np.sum(x != 0)
        self.P = P
        self.a = self.M / self.N
        self.M_p = int(self.M / self.P)
        self.A_p = A.reshape(P, self.M_p, self.N)
        self.x = x
        if type(noise) is int:
            self.noise = [noise] * self.P
        elif type(noise).__module__ == 'numpy':
            self.noise = noise.reshape(P, self.M_p, 1)
        else :
            raise ValueError
        self.s = np.zeros((self.N, 1))
        self.mse = np.array([None])
        self.communication_cost = np.array([])
        self.r2 = np.zeros(self.P)
        self.tau_p = np.zeros(self.P)
        self.v_p = np.zeros(self.P)
        self.v = [None]
        self.booleans = (x == 0)
        self.mse_non_zero = np.array([None])
        self.mse_zero = np.array([None])

    def _add_mse(self):
        mse = np.linalg.norm(self.s - self.x)**2 / self.N
        self.mse = np.append(self.mse, mse)

        sum_4_zero = 0
        sum_4_non_zero = 0
        for i in self.booleans:
            if i:
                sum_4_zero += self.s[i][0]**2
            elif not i:
                sum_4_non_zero += (self.s[i][0] - self.x[i][0])**2
            else:
                raise ValueError("Not Correct Value")
        self.mse_zero = np.append(self.mse_zero, sum_4_zero / (self.N - self.K))
        self.mse_non_zero = np.append(self.mse_non_zero, sum_4_non_zero / self.K)

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
        ite = np.arange(0, np.shape(self.mse)[0], 1)
        plt.xticks(ite)
        result = np.array([np.log10(mse) if mse is not None else None for mse in self.mse])
        plt.plot(result)
        se = np.array([np.log10(v) if v is not None else None for v in self.v])
        plt.scatter(ite, se, c='red')
        plt.grid()
