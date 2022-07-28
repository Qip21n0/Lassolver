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
        self.zero_index = x == 0
        self.non_zero_index = x != 0
        self.mse_zero = np.array([None])
        self.mse_non_zero = np.array([None])
        self.mse_diff_zero = np.array([None])
        self.mse_diff_non_zero = np.array([None])
        self.mse_hist_bins = []

    def _add_mse(self):
        mse = np.linalg.norm(self.s - self.x)**2 / self.N
        self.mse = np.append(self.mse, mse)

        sum_4_zero = 0
        sum_4_non_zero = 0
        for k, v in enumerate(self.zero_index):
            if v:
                sum_4_zero += self._square_error_4_component(k)
            elif not v:
                sum_4_non_zero += self._square_error_4_component(k)
            else:
                raise ValueError("Not Correct Value")
        self.mse_zero = np.append(self.mse_zero, sum_4_zero[0] / (self.N - self.K))
        self.mse_non_zero = np.append(self.mse_non_zero, sum_4_non_zero[0] / self.K)

    def _square_error_4_component(self, i):
        return (self.s[i] - self.x[i])**2

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

    def _inspect_b_w(self, diff_b_w):
        diff_zero_index = diff_b_w == 0
        sum_4_diff_zero = 0
        sum_4_diff_non_zero = 0
        for k, v in enumerate(diff_zero_index):
            if v:
                sum_4_diff_zero += self._square_error_4_component(k)
            elif not v:
                sum_4_diff_non_zero += self._square_error_4_component(k)
            else:
                raise ValueError("Not Correct Value")
        num_diff_zero = np.sum(diff_zero_index)
        self.mse_diff_zero = np.append(self.mse_diff_zero, sum_4_diff_zero[0] / num_diff_zero)
        self.mse_diff_non_zero = np.append(self.mse_diff_non_zero, sum_4_diff_non_zero[0] / (self.N - num_diff_zero))

        size = 50
        hist, bins = np.histogram(diff_b_w, bins=size) # hist: R^50, bins: R^51
        index_4_hist = np.digitize(diff_b_w, bins) - 1 # index_4_hist: R^N (0~50), diff_b_w: R^N
        mse_hist_bins = np.zeros((3, 3, size+1))
        """
        mse_hist_bins
            1-dim  0: all, 1: zero, 2: non zero
            2-dim  0: mse, 1: hist, 3: bins
            3-dim  R^51
        """
        mse_hist_bins[0, 1] = np.append(hist.copy(), 0) # 1: hist for all
        hist_zero, _ = np.histogram(diff_b_w[self.zero_index], bins=bins)
        mse_hist_bins[1, 1] = np.append(hist_zero.copy(), 0) # 1: hist for zero
        hist_non_zero, _ = np.histogram(diff_b_w[self.non_zero_index], bins=bins)
        mse_hist_bins[2, 1] = np.append(hist_non_zero.copy(), 0) # 1: hist for non zero

        mse_hist_bins[:, 2] = bins.copy() # 2: bins
        for i in range(self.N):
            mse = self._square_error_4_component(i)
            j = index_4_hist[i]
            j = j if j != size else j-1

            mse_hist_bins[0, 0, j] += mse
            if self.zero_index[i]:
                mse_hist_bins[1, 0, j] += mse
            elif self.non_zero_index[i]:
                mse_hist_bins[2, 0, j] += mse
            else:
                raise ValueError("Not Correct Value")

        for i in range(size):
            for j in range(3):
                if mse_hist_bins[j, 1, i] != 0: # hist != 0
                    mse_hist_bins[j, 0, i] /= mse_hist_bins[j, 1, i] # mse /= hist
                elif mse_hist_bins[j, 0, i] != 0: # mse != 0
                    print("\033[33mERROR\033[0m: mse = ", mse_hist_bins[j, 0, i], "hist = ", mse_hist_bins[j, 1, i])

        self.mse_hist_bins.append(mse_hist_bins)