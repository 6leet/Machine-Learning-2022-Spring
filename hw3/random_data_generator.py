import numpy as np
import math
import matplotlib.pyplot as plt

def standard_gaussian(size=1): # Box-Muller Method
    U = np.random.uniform(size=size)
    V = np.random.uniform(size=size)
    X = [math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v) for u, v in zip(U, V)]
    return X

def gaussian(mean=0, var=1, size=1):
    std = math.sqrt(var)
    return [mean + x * std for x in standard_gaussian(size)]

def polynomial_basis_linear_model(n, a, w, size=1): # n: basis number
    E = gaussian(0, a, size=size)
    X = np.random.uniform(-1, 1, size=size)
    Y = []
    for e, x in zip(E, X):
        _x = 1
        y = 0
        for _w in w:
            y += _w * _x
            _x *= x
        y += e
        Y.append(y)
    return X, Y
    

