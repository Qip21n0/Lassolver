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