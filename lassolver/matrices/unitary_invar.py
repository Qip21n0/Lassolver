import numpy as np
from scipy.stats import ortho_group

class UnitaryInvar:
    def __init__(self, M, N, connum):
        self.kappa = connum
        self.r = kappa**(1/M)
        self.sv = self.singular_value(self.kappa, self.r, M, N)
        self.S = np.hstack((np.diag(self.sv), np.zeros((M, N-M)))
        self.V = ortho_group(M)
        self.U = ortho_group(N)
        self.A = self.V @ self.S @ self.U.T

    def singular_value(self, kappa, r, M, N):
        sv = np.array([N * (1 - 1/r) / (1 - 1/kappa) if kappa != 1 else N/M])
        for i in range(M-1):
            sv = np.append(sv, sv[-1] / r)
        return sv
