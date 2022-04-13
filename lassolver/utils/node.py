import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np



class Node:
	def __init__(self, A, x, snr):
		self.M, self.N = A.shape
		self.A = A.copy()
		self.x = x.copy()

		Ax = A @ x
		SNRdB = 10**(0.1 * snr)
		self.sigma = np.linalg.norm(Ax) ** 2 / SNRdB
		self.n = np.random.normal(0, self.sigma**0.5, (self.M, 1))

		self.y = Ax + self.n