from lassolver.utils.func import *
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
        self.tau = [None]
        self.v = [None]
        self.zeros = x == 0
        self.non_zeros = x != 0
        self.confusion_matrix = []
        self.evaluation_index = {"accuracy": np.array([]), 
                                 "precision": np.array([]), 
                                 "recall": np.array([]), 
                                 "F1": np.array([]), 
                                 "MCC": np.array([])}
        
        self.mse_zero = np.array([None])
        self.mse_non_zero = np.array([None])
        self.confusion_matrix_4_oamp = []
        self.mse_diff_zero = np.array([None])
        self.mse_diff_non_zero = np.array([None])
        self.s_history_4_diff_non_zero = []
        self.mse_hist_bins = []
        self.w_b_z_history = []


    def _add_mse(self):
        mse = np.linalg.norm(self.s - self.x)**2 / self.N
        self.mse = np.append(self.mse, mse)

        sum_4_zero = 0
        sum_4_non_zero = 0
        for k, v in enumerate(self.zeros):
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


    def _make_confusion_matrix(self, b_w):
        diff_zeros = b_w == 0
        diff_non_zeros = b_w != 0

        index = {}
        index["TP"] = self.zeros & diff_non_zeros # x = 0 & diff != 0
        index["FP"] = self.non_zeros & diff_non_zeros # x != 0 & diff != 0
        index["FN"] = self.zeros & diff_zeros # x = 0 & diff = 0
        index["TN"] = self.non_zeros & diff_zeros # x != 0 & diff = 0

        mse_quantity_table = np.zeros((2, 3, 3)) # 0: MSE or Quantity, 1: diff is zero or not, 2: x is zero or not
        for i, key in enumerate(index):
            for j, v in enumerate(index[key]):
                if v:
                    mse_quantity_table[0, i//2, i%2] += self._square_error_4_component(j)

            mse_quantity_table[1, i//2, i%2] = np.sum(index[key]) # Quantity
            mse_quantity_table[0, i//2, i%2] /= mse_quantity_table[1, i//2, i%2] # MSE

        # diff is zero or not
        for j, v in enumerate(diff_non_zeros):
            if v:
                mse_quantity_table[0, 0, 2] += self._square_error_4_component(j) # diff != 0
            elif not v:
                mse_quantity_table[0, 1, 2] += self._square_error_4_component(j) # diff = 0
            else:
                raise ValueError("Not Correct Value")
        mse_quantity_table[1, 0, 2] = np.sum(diff_non_zeros)
        mse_quantity_table[1, 1, 2] = np.sum(diff_zeros)
        mse_quantity_table[0, 0, 2] /= mse_quantity_table[1, 0, 2]
        mse_quantity_table[0, 1, 2] /= mse_quantity_table[1, 1, 2]

        # x is zero or not
        for j, v in enumerate(self.zeros):
            if v:
                mse_quantity_table[0, 2, 0] += self._square_error_4_component(j) # x = 0
            elif not v:
                mse_quantity_table[0, 2, 1] += self._square_error_4_component(j) # x != 0
            else:
                raise ValueError("Not Correct Value")
        mse_quantity_table[1, 2, 0] = self.N - self.K
        mse_quantity_table[1, 2, 1] = self.K
        mse_quantity_table[0, 2, 0] /= mse_quantity_table[1, 2, 0]
        mse_quantity_table[0, 2, 1] /= mse_quantity_table[1, 2, 1]

        # MSE and Quantity
        mse_quantity_table[0, 2, 2] = np.linalg.norm(self.s - self.x)**2 / self.N
        mse_quantity_table[1, 2, 2] = self.N

        self.confusion_matrix.append(mse_quantity_table)


    def _evaluate_performance(self):
        confusion_matrix = self.confusion_matrix[-1][1].copy() # Quantity

        # accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / self.N

        # precision = TP / (TP + FP)
        precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
        
        # recall = TP / (TP + FN)
        recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
        
        # F1 = 2TP / (2TP + FP + FN)
        F1 = 2 / (1/recall + 1/precision)
        
        # MCC = (TP * TN - FP * FN) / (()()())^1/2
        top = confusion_matrix[0, 0] * confusion_matrix[1, 1] - confusion_matrix[0, 1] * confusion_matrix[1, 0]
        bottom = (confusion_matrix[0, 0]+confusion_matrix[0, 1]) * (confusion_matrix[0, 0]+confusion_matrix[1, 0]) * (confusion_matrix[1, 1]+confusion_matrix[0, 1]) * (confusion_matrix[1, 1]+confusion_matrix[1, 0])
        MCC = top / bottom**0.5

        self.evaluation_index["accuracy"] = np.append(self.evaluation_index["accuracy"], accuracy)
        self.evaluation_index["precision"] = np.append(self.evaluation_index["precision"], precision)
        self.evaluation_index["recall"] = np.append(self.evaluation_index["recall"], recall)
        self.evaluation_index["F1"] = np.append(self.evaluation_index["F1"], F1)
        self.evaluation_index["MCC"] = np.append(self.evaluation_index["MCC"], MCC)

    
    def _make_confusion_matrix_4_oamp(self, C, w, b_w):
        s_oamp = C * df(np.sum(w, axis=0), np.sum(self.tau_p)**0.5)
        def square_error_4_component(i):
            return (s_oamp[i] - self.x[i])**2
        
        diff_zeros = b_w == 0
        diff_non_zeros = b_w != 0

        index = {}
        index["TP"] = self.zeros & diff_non_zeros # x = 0 & diff != 0
        index["FP"] = self.non_zeros & diff_non_zeros # x != 0 & diff != 0
        index["FN"] = self.zeros & diff_zeros # x = 0 & diff = 0
        index["TN"] = self.non_zeros & diff_zeros # x != 0 & diff = 0

        mse_quantity_table = np.zeros((2, 3, 3)) # 0: MSE or Quantity, 1: diff is zero or not, 2: x is zero or not
        for i, key in enumerate(index):
            for j, v in enumerate(index[key]):
                if v:
                    mse_quantity_table[0, i//2, i%2] += square_error_4_component(j)

            mse_quantity_table[1, i//2, i%2] = np.sum(index[key]) # Quantity
            mse_quantity_table[0, i//2, i%2] /= mse_quantity_table[1, i//2, i%2] # MSE

        # diff is zero or not
        for j, v in enumerate(diff_non_zeros):
            if v:
                mse_quantity_table[0, 0, 2] += square_error_4_component(j) # diff != 0
            elif not v:
                mse_quantity_table[0, 1, 2] += square_error_4_component(j) # diff = 0
            else:
                raise ValueError("Not Correct Value")
        mse_quantity_table[1, 0, 2] = np.sum(diff_non_zeros)
        mse_quantity_table[1, 1, 2] = np.sum(diff_zeros)
        mse_quantity_table[0, 0, 2] /= mse_quantity_table[1, 0, 2]
        mse_quantity_table[0, 1, 2] /= mse_quantity_table[1, 1, 2]

        # x is zero or not
        for j, v in enumerate(self.zeros):
            if v:
                mse_quantity_table[0, 2, 0] += square_error_4_component(j) # x = 0
            elif not v:
                mse_quantity_table[0, 2, 1] += square_error_4_component(j) # x != 0
            else:
                raise ValueError("Not Correct Value")
        mse_quantity_table[1, 2, 0] = self.N - self.K
        mse_quantity_table[1, 2, 1] = self.K
        mse_quantity_table[0, 2, 0] /= mse_quantity_table[1, 2, 0]
        mse_quantity_table[0, 2, 1] /= mse_quantity_table[1, 2, 1]

        # MSE and Quantity
        mse_quantity_table[0, 2, 2] = np.linalg.norm(s_oamp - self.x)**2 / self.N
        mse_quantity_table[1, 2, 2] = self.N

        self.confusion_matrix_4_oamp.append(mse_quantity_table)

    
    def _add_s_history_4_diff_non_zero(self, diff_b_w):
        diff_non_zero_index = diff_b_w != 0
        s = np.array([])
        for k, v in enumerate(diff_non_zero_index):
            if v:
                s = np.append(s, self.s[k])
            else:
                s = np.append(s, None)
        self.s_history_4_diff_non_zero.append(s)

    
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

        size = 100
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
        hist_zero, _ = np.histogram(diff_b_w[self.zeros], bins=bins)
        mse_hist_bins[1, 1] = np.append(hist_zero.copy(), 0) # 1: hist for zero
        hist_non_zero, _ = np.histogram(diff_b_w[self.non_zeros], bins=bins)
        mse_hist_bins[2, 1] = np.append(hist_non_zero.copy(), 0) # 1: hist for non zero

        mse_hist_bins[:, 2] = bins.copy() # 2: bins
        for i in range(self.N):
            mse = self._square_error_4_component(i)
            j = index_4_hist[i]
            j = j if j != size else j-1

            mse_hist_bins[0, 0, j] += mse
            if self.zeros[i]:
                mse_hist_bins[1, 0, j] += mse
            elif self.non_zeros[i]:
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


    def _add_w_b_z_hisory(self, w_p, b, z):
        w = np.sum(w_p, axis=0)
        diff_b_w = b - w
        z = np.array(z)

        diff_zeros = diff_b_w == 0
        diff_non_zeros = diff_b_w != 0

        index = {}
        index["TP"] = self.zeros & diff_non_zeros # x = 0 & diff != 0
        index["FP"] = self.non_zeros & diff_non_zeros # x != 0 & diff != 0
        index["FN"] = self.zeros & diff_zeros # x = 0 & diff = 0
        index["TN"] = self.non_zeros & diff_zeros # x != 0 & diff = 0

        history = {"TP": np.empty((3, self.N)),
                   "FP": np.empty((3, self.N)), 
                   "FN": np.empty((3, self.N)), 
                   "TN": np.empty((3, self.N))}
        
        for key in index:
            for j, v in enumerate(index[key]):
                if v:
                    history[key][0][j] = w[j]
                    history[key][1][j] = b[j]
                    history[key][2][j] = z[j]
                else:
                    history[key][0][j] = None
                    history[key][1][j] = None
                    history[key][2][j] = None

        self.w_b_z_history.append(history)