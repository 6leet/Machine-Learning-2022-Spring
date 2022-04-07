import random_data_generator as random
import matrix as mat

def regression(b, n, a, w):
    var = a
    a = 1 / a
    X, Y = random.polynomial_basis_linear_model(n, a, w, 1)
    X = mat.Matrix([X])
    Y = mat.Matrix([Y])
    aXtX = X.transpose().product(X).mult(a)
    covmat = aXtX.add(mat.Matrix.identity(aXtX.w).mult(b))
    mean = covmat.inverse().product(X.transpose()).product(Y).mult(a)

    mean.show()
    covmat.show()

regression(1, 4, 1, [1, 2, 3, 4])
