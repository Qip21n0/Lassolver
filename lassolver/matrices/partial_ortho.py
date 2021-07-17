import numpy as np
from scipy.stats import ortho_group
from scipy.fftpack import dct
from scipy.linalg import hadamard
from lassolver.metrices.base import Base

class PartialOrtho(Base):
    def __init__(self, M, N, ord='Haar'):
        super().__init__(M, N)
        self.I = np.eye(N)
        s = np.random.randint(0, N, M)
        self.S = np.array([self.I[i] for i in s])
        self.U = self.__set_U(ord)
        self.A = (N/M)**0.5 * self.S @ self.U.T

    def __set_U(self, ord):
        if ord == 'Haar':
            return ortho_group.rvs(self.N)
        elif ord == 'DCT':
            return dct(self.I, axis=0, norm='ortho')
        elif ord == 'Hadamard':
            return hadamard(self.N)
        else :
            raise NameError("Enter one of Haar, DCT, or Hadamard for the argument (ord)")
