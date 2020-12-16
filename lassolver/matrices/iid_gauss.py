import numpy as np

class iidGaussian():
    def __init__(self, row, column, mean=0, var=1):
        self.M = row
        self.N = column
        self.mean = mean
        self.var = var
        self.A = self.set_matrix(M, N, mean, var)

    def set_matrix(self, M, N, m, v):
        """
        Return i.i.d(independent and identically distributed) Gaussian Matrix
        """
        return np.random.normal(m, v**0.5, (M, N))
