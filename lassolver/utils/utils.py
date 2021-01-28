import time
from functools import wraps
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

def distri(w, t, ite_max, ite):
    N = w.shape[0]
    w_ = np.array([0])
    for i in range(N):
        if np.abs(w[i]) < t:
            w_ = np.append(w_, w[i]/t)
    print("ite: {}, m: {}, v: {}".format(ite+1, np.mean(w_[1:]), np.var(w_[1:])))
    plt.subplot(ite_max, 1, ite+1)
    plt.hist(w_[1:], bins=50, density=True)