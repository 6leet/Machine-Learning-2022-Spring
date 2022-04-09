import random_data_generator as random
import matrix as mat
from matplotlib import pyplot as plt
import numpy as np

def plot_poly_by_coeffs(X, coeffs, var):
    Y = 0
    for i in range(len(coeffs)):
        Y += coeffs[i]*X**i
    Y += var

    yticks = range(-15, 20, 5)
    plt.xlim(X[0], X[-1])
    plt.ylim(-15, 20)
    plt.yticks(yticks)

    if var == 0:
        plt.plot(X, Y, 'k')
    else:
        plt.plot(X, Y, 'r')

def plot_poly_by_predict(X, mean_vec, covariance_mat, n, a, var_c):
    Y = []
    for x in X:
        phi_x = get_design_matrix(x, n)
        phi_x = mat.Matrix([phi_x])
        predict_var = mat.Matrix([[1 / a]]).add(phi_x.product(covariance_mat).product(phi_x.transpose())).access(0, 0)
        Y.append(mean_vec.transpose().product(phi_x.transpose()).access(0, 0) + predict_var * var_c)

    yticks = range(-15, 20, 5)
    plt.xlim(X[0], X[-1])
    plt.ylim(-15, 20)
    plt.yticks(yticks)

    if var_c == 0:
        plt.plot(X, Y, 'k')
    else:
        plt.plot(X, Y, 'r')

def plot_ground(coeffs, var):
    plot_poly_by_coeffs(np.linspace(-2, 2, 100), coeffs, 0)
    plot_poly_by_coeffs(np.linspace(-2, 2, 100), coeffs, var)
    plot_poly_by_coeffs(np.linspace(-2, 2, 100), coeffs, -var)

def plot_predict(mean_vec, covariance_mat, n, a):
    plot_poly_by_predict(np.linspace(-2, 2, 100), mean_vec, covariance_mat, n, a, 0)
    plot_poly_by_predict(np.linspace(-2, 2, 100), mean_vec, covariance_mat, n, a, 1)
    plot_poly_by_predict(np.linspace(-2, 2, 100), mean_vec, covariance_mat, n, a, -1)

def plot_points(X, Y):
    for x, y in zip(X, Y):
        plt.plot(x, y, 'b.')

def get_design_matrix(x: float, n):
    _x = 1
    phi_x = []
    for i in range(n):
        phi_x.append(_x)
        _x *= x
    return phi_x

def regression(b, n, a, w):
    var = a
    a = 1 / a
    i = 0
    xs = []
    phi_xs = [] 
    ys = []
    S = mat.Matrix.identity(n).mult(0)
    m = mat.Matrix([[0] * n]).transpose()
    last_predict_var = 0
    while True:
        _x, _y = random.polynomial_basis_linear_model(n, var, w, 1)
        x, y = _x[0], _y[0]
        print('Add data point (' + str(x) + ', ' + str(y) + ')')
        print()

        xs.append(x)
        phi_x = get_design_matrix(x, n)
        phi_xs.append(phi_x)
        ys.append(y)

        phi_X = mat.Matrix(phi_xs)
        Y = mat.Matrix([ys]).transpose()

        aXtX  = phi_X.transpose().product(phi_X).mult(a)
        precision_mat = aXtX.add(mat.Matrix.identity(aXtX.w).mult(b))
        covariance_mat = precision_mat.inverse()
        mean_vec = covariance_mat.product(phi_X.transpose().product(Y).mult(a).add(S.product(m)))

        print('Posterior mean:')
        mean_vec.show()
        print()

        print('Posterior variance:')
        covariance_mat.show()
        print()

        phi_x = mat.Matrix([phi_x])
        predict_mean = mean_vec.transpose().product(phi_x.transpose()).access(0, 0)
        predict_var = mat.Matrix([[1 / a]]).add(phi_x.product(covariance_mat).product(phi_x.transpose())).access(0, 0)
        print('Predictive distribution ~ N(' + str(predict_mean) + ', ' + str(predict_var) + ')')
        print()

        if abs(last_predict_var - predict_var) < 1e-6:
            plt.subplot(222)
            plt.title('Predict result')
            plot_points(xs, ys)
            plot_predict(mean_vec, covariance_mat, n, a)
            return

        if i == 0:
            S = precision_mat
            m = mean_vec

        if i == 10:
            plt.subplot(223)
            plt.title('After 10 incomes')
            plot_points(xs, ys)
            plot_predict(mean_vec, covariance_mat, n, a)
        elif i == 50:
            plt.subplot(224)
            plt.title('After 50 incomes')
            plot_points(xs, ys)
            plot_predict(mean_vec, covariance_mat, n, a)

        last_predict_var = predict_var

        i += 1
b = 1
n = 3
a = 3
w = [1, 2, 3]
plt.subplot(221)
plt.title('Ground truth')
plot_ground(w, a)
regression(b, n, a, w)
plt.show()