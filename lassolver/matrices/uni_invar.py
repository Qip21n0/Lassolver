import numpy as np
from scipy.stats import ortho_group
from lassolver.matrices.base import Base

class UniInvar(Base):
    def __init__(self, M, N, cond):
        super().__init__(M, N)
        self.kappa = cond
        self.r = cond**(1/M)
        self.sv = self.singular_value()
        self.Sigma = np.hstack((np.diag(self.sv), np.zeros((M, N-M))))
        self.V = ortho_group.rvs(M)
        self.U = ortho_group.rvs(N)
        self.A = self.V @ self.Sigma @ self.U.T

    def singular_value(self):
        start = self.N * (1 - 1/self.r) / (1 - 1/self.kappa) if self.kappa != 1 else self.N/self.M
        sv = np.array([start])
        for i in range(self.M-1):
            sv = np.append(sv, sv[-1] / self.r)
        return sv

    def change_cond(self, cond):
        self.kappa = cond
        self.r = cond**(1/self.M)
        self.sv = self.singular_value()
        self.Sigma = np.hstack((np.diag(self.sv), np.zeros((self.M, self.N - self.M))))
        self.A = self.V @ self.Sigma @ self.U.T
