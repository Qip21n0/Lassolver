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
		self.s = jnp.zeros(self.N, dtype=jnp.float32)
		self.s_history = []

		self.AT = self.A.T
		self.mse = [None]


	def _add_mse(self):
		mse = np.linalg.norm(self.s - self.x)**2 / self.N
		self.mse.append(mse)


	def show_result(self):
		print("final mse: {}".format(self.mse[-1]))
		plt.figure(figsize=(16, 4))

		plt.subplot(121)
		plt.plot(self.x.real)
		plt.plot(self.s.real)
		plt.grid()
		
		plt.subplot(122)
		plt.xlabel('iteration')
		plt.ylabel('MSE[log10]')
		iter = np.shape(self.mse)[0]
		plt.xticks(np.arange(0, iter, 1))
		result = np.array([np.log10(mse) if mse is not None else None for mse in self.mse])
		plt.plot(result)
		plt.grid()