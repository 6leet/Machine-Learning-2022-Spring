import random_data_generator as random
import matrix as mat
from matplotlib import pyplot as plt
import numpy as np

def plot_poly_var(x, coeffs, var):
    y = 0
    for i in range(len(coeffs)):
        y += coeffs[i]*x**i
    
    y += var
    plt.plot(x, y)
    # plt.show()

def plot_poly_predict(X, mean, covmat, a, n, c):
    y = []
    for x in X:
        fiX = get_design_matrix(x, n)
        fiX = mat.Matrix([fiX])
        var = mat.Matrix([[1 / a]]).add(fiX.product(covmat).product(fiX.transpose())).access(0, 0)
        y.append(mean.transpose().product(fiX.transpose()).access(0, 0) + var * c)

    yticks = range(-20, 20, 5)
    plt.xlim(X[0], X[-1])
    plt.ylim(-20, 20)
    plt.yticks(yticks)
    plt.plot(X, y)
    # plt.plot(X, y)

def get_design_matrix(x, n):
    _x = 1
    X = []
    for i in range(n):
        X.append(_x)
        _x *= x
    return X

def regression(b, n, a, w):
    var = a
    a = 1 / a
    i = 0
    S = mat.Matrix.identity(n).mult(0)
    m = mat.Matrix([[0] * n]).transpose()
    arrX = []
    arrY = []
    N = 1000
    while i < N:
        _X, _Y = random.polynomial_basis_linear_model(n, a, w, 1)
        print('Add data point (' + str(_X[0]) + ', ' + str(_Y[0]) + ')')
        # _X, _Y = [-0.64152], [0.19039]
        fiX = get_design_matrix(_X[0], n)
        arrX.append(fiX)
        X = mat.Matrix(arrX)
        arrY.append(_Y[0])
        # X.show()
        Y = mat.Matrix([arrY]).transpose()
        aXtX = X.transpose().product(X).mult(a)
        premat = aXtX.add(mat.Matrix.identity(aXtX.w).mult(b))
        covmat = premat.inverse()
        mean = covmat.product(X.transpose().product(Y).mult(a).add(S.product(m)))

        print('Posterior mean:')
        mean.show()
        print('Posterior variance:')
        covmat.show()

        if i == 10 or i == 50 or i == N - 1:
            # arrmean = mean.transpose().mat()[0]
            # plot_poly_cov(np.linspace(-2, 2, 100), arrmean, [0, 0, 0, 0])
            # print([covmat.access(i, i) for i in range(n)])
            # plot_poly_cov(np.linspace(-2, 2, 100), arrmean, [covmat.access(i, i) for i in range(n)])
            # plot_poly_cov(np.linspace(-2, 2, 100), arrmean, [-covmat.access(i, i) for i in range(n)])
            plot_poly_predict(np.linspace(-2, 2, 100), mean, covmat, a, n, 0)
            plot_poly_predict(np.linspace(-2, 2, 100), mean, covmat, a, n, 1)
            plot_poly_predict(np.linspace(-2, 2, 100), mean, covmat, a, n, -1)
            plt.show()


        fiX = mat.Matrix([fiX])

        print('Predictive distribution ~')
        mean.transpose().product(fiX.transpose()).show()
        mat.Matrix([[1 / a]]).add(fiX.product(covmat).product(fiX.transpose())).show()
        if i == 0:
            S = premat
            m = mean

        i += 1

regression(1, 3, 3, [1, 2, 3])
plot_poly_var(np.linspace(-2, 2, 100), [1, 2, 3, 4], 0)
plot_poly_var(np.linspace(-2, 2, 100), [1, 2, 3, 4], 1)
plot_poly_var(np.linspace(-2, 2, 100), [1, 2, 3, 4], -1)
plt.show()

