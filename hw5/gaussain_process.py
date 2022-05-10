import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

def preprocess(filename):
    X = []
    Y = []
    with open(filename, 'r') as file:
        for line in file:
            x, y = line.split()
            X.append(float(x))
            Y.append(float(y))

    return np.array(X), np.array(Y)

def rational_quadratic_kernel(X1, X2, sigma, alpha, l):
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    
    dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
    return (sigma ** 2) * (1 + dist / (2 * alpha * (l ** 2))) ** (-alpha)

def negative_log_likelihood(params, X, Y, beta=5):
    sigma, alpha, l = params[0], params[1], params[2]
    k = rational_quadratic_kernel(X, X, sigma, alpha, l)
    C = k + np.identity(len(X)) * (1 / beta)
    Cinv = np.linalg.inv(C)
    return np.log(np.linalg.det(C)) + Y.T.dot(Cinv).dot(Y) + len(X) * np.log(2 * math.pi)

def predict(X, Y, test_X, sigma, alpha, l, beta=5):
    k = rational_quadratic_kernel(X, X, sigma, alpha, l)
    Cinv = np.linalg.inv(k + np.identity(len(X)) * (1 / beta))
    k_s = rational_quadratic_kernel(X, test_X, sigma, alpha, l)
    k_ss = rational_quadratic_kernel(test_X, test_X, sigma, alpha, l)

    mean = k_s.T.dot(Cinv).dot(Y)
    cov = k_ss + np.identity(len(test_X)) * (1 / beta) - k_s.T.dot(Cinv).dot(k_s)

    return mean, cov

def visualize(X, Y, test_X, mean, cov, sigma, alpha, l):
    for x, y in zip(X, Y):
        plt.plot(x, y, 'k.')

    plt.plot(test_X, mean, 'b')

    interval = 1.96 * np.sqrt(np.diag(cov))
    plt.plot(test_X, mean + interval, 'r')
    plt.plot(test_X, mean - interval, 'r')
    plt.fill_between(test_X, mean + interval, mean - interval, color='r', alpha=0.2)

    plt.xlim(-60, 60)

    title = 'sigma = {:.4f}, alpha = {:.4f}, l = {:.4f}'.format(sigma, alpha, l)
    plt.title(title)
    plt.show()

def gaussian_process(filename, optimize=True):
    X, Y = preprocess(filename)
    test_X = np.linspace(-60, 60, 1000)

    sigma = 1
    alpha = 1
    l = 1
    if optimize:
        bound = (1e-6, 1e6)
        sigma, alpha, l = minimize(negative_log_likelihood, [sigma, alpha, l], bounds=(bound, bound, bound), args=(X, Y)).x

    mean, cov = predict(X, Y, test_X, sigma, alpha, l)
    visualize(X, Y, test_X, mean, cov, sigma, alpha, l)

gaussian_process('ML_HW05/data/input.data', True)