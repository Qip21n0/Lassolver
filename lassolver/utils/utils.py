import time
from functools import wraps
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

def ftime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        fin = time.time()
        print("{}: {}[s]".format(func.__name__, fin - start))
        return result
    return wrapper

def details(inst):
    print("class: {}\n".format(inst.__class__.__name__))
    for key, value in inst.__dict__.items():
        print("member: {}".format(key))
        print("type: {}".format(type(value)))
        print("shape: {}".format(np.shape(value)))
        print(value)
        print("")

def plt_CC(N, P, T, cc_dict):
	x_step = np.arange(0, T+1, 5)
	y_step = np.arange(0, 1, 0.1)
	standard = N * (P-1)

	plt.xlabel("iteration")
	plt.ylabel("Communication Cost")
	plt.xticks(x_step)
	plt.yticks(y_step)
	plt.ylim(0, 1)

	for k, v in cc_dict.items():
		cc = v.copy()
		cc /= standard
		plt.plot(cc, label=k)
	plt.legend()
	plt.grid()


def plt_MSE(T, mse_dict):
	step = np.arange(0, T+1, 5)

	plt.xlabel("iteration")
	plt.ylabel("MSE")
	plt.xticks(step)
	plt.ylim(1e-3, 1e+1)
	plt.yscale('log')

	for k, v in mse_dict.items():
		plt.plot(v, label=k)
	plt.legend()
	plt.grid()


def plt_detail(N, P, T, w, tau):
    plt.figure(figsize=(7*P, 6*T))
    I = np.ones(N)
    shita = 1
    for t in range(T):
        beta = np.sum(tau[t])**0.5
        for p in range(P):
            plt.subplot(T, P, P*t + p+1)
            plt.plot(w[t][p])
            plt.plot(I * tau[t][p]**0.5 * shita, color="r")
            plt.plot(-I * tau[t][p]**0.5 * shita, color="r")