import numpy as np
import matplotlib.pyplot as plt

class Base:
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.A = np.hstack((np.eye(M), np.zeros((M, N-M))))

    def matrix(self):
        print("row: {}".format(self.M))
        print("column: {}".format(self.N))
        print("mean: {}".format(self.A.mean()))
        print("variance: {}".format(self.A.var()))

    def show_hist(self):
        A = self.A.reshape(self.M * self.N)
        plt.hist(A, bins=50, density=True)