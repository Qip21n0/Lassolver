import numpy as np

class iidGaussian:
    def __init__(self, M, N, m=0, v=1):
        self.A = self.set_matrix(M, N, m, v)

    def set_matrix(self, row, column, mean, var):
        """
        Return i.i.d(independent and identically distributed) Gaussian Matrix
        """
        return np.random.normal(mean, var**0.5, (row, column))
