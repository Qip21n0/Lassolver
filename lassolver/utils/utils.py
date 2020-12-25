import numpy as np
import time
from functools import wraps

def func_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        fin = time.time()
        print("{}: {}[s]".format(func.__name__, fin - start))
        return result
    return wrapper


def details(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("function name: {}".format(func.__name__))
        print("-- arguments --")
        for arg in args:
            print("{}".format(arg))
            print("type: {}".format(type(arg)))
            #print("shape: {}".format(np.shape(arg)))
        for key in kwargs:
            print("{}: {}".format(key, kwargs[key]))
            print("type: {}".format(type(kwargs[key])))
        result = func(*args, **kwargs)
        for value in dir(func):
            print(value)
        return result
    return wrapper

@details
def xxx(x, y):
    a = 2
    return a * (x + y)

xxx(3, 6)