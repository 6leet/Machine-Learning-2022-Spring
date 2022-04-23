from sympy import true
import random_data_generator as random
from matplotlib import pyplot as plt
import matrix as mat
import math

def plot_data_points(D, cluster, subplot):
    plt.subplot(subplot)
    color = ['r', 'b']
    for y, d in zip(cluster, D):
        y = round(y)
        plt.plot(d[0], d[1], color[y] + '.')

def print_confusion(confusion_matrix):
    print('Confusion Matrix:')
    s = 'predict cluster 1 predict cluster 2'
    s1 = 'Is cluster 1'
    s2 = 'Is cluster 2'
    print(s.rjust(len(s) + len(s1)))
    print(s1 + str(confusion_matrix[0][0]).center(len(s) // 2) + str(confusion_matrix[0][1]).center(len(s) // 2))
    print(s2 + str(confusion_matrix[1][0]).center(len(s) // 2) + str(confusion_matrix[1][1]).center(len(s) // 2))

def print_result(method, W, confusion_matrix):
    print(method + ':\n')
    print('W:')
    W.transpose().show()

    print()
    print_confusion(confusion_matrix)

    print()
    print('Sensitivity (Successfully predict cluster 1):', confusion_matrix[0][0] / sum(confusion_matrix[0]))
    print('Specificity (Successfully predict cluster 2):', confusion_matrix[1][1] / sum(confusion_matrix[1]))

    print()
    print('-' * 50)

def generate_data_points(N, mean_x, var_x, mean_y, var_y):
    X = random.gaussian(mean_x, var_x, N)
    Y = random.gaussian(mean_y, var_y, N)
    D = [[x, y] for x, y in zip(X, Y)]
    return D

def generate_ground_truth(N, mxs, vxs, mys, vys):
    D = []
    ground_truth = []
    for i in range(len(mxs)):
        D.extend(generate_data_points(N, mxs[i], vxs[i], mys[i], vys[i]))
        ground_truth.extend([i] * N)

    for i in range(len(D)):
        D[i].append(1)
    
    return D, ground_truth

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def predict(W: mat.Matrix, D: mat.Matrix):
    Z = D.product(W.transpose())

    result = []
    for i in range(Z.h):
        for j in range(Z.w):
            result.append(sigmoid(Z.access(i, j)))

    return result

def get_confusion_matrix(result, ground_truth):
    confusion_matrix = [[0, 0], [0, 0]]
    for r, g in zip(result, ground_truth):
        r = round(r)
        confusion_matrix[g][r] += 1

    return confusion_matrix

def gradient_descent(W, D, ground_truth, lr):
    W = mat.Matrix([W])
    At = mat.Matrix(D).transpose()

    i = 0
    while i < 1000:
        i += 1

        Z = W.product(At).mat()[0]
        delta = [sigmoid(z) - y for y, z in zip(ground_truth, Z)]
        delta = mat.Matrix([delta]).transpose()

        W = W.add(At.product(delta).transpose().mult(-1 * lr))

    return W

def hessian(W, D):
    At = mat.Matrix(D).transpose()
    H = [[0] * W.w for _ in range(W.w)]

    for j in range(W.w):
        for k in range(W.w):
            Z = W.product(At).mat()[0]
            for z, d in zip(Z, D):
                try:
                    z = math.exp(-z)
                    H[j][k] += (d[j] * d[k] * z) / pow((1 + z), 2)
                except OverflowError:
                    z = 1e5
                    H[j][k] += (d[j] * d[k] * z) / pow((1 + z), 2)
    return H 

def newton_method(W, D, ground_truth, lr):

    W = mat.Matrix([W])
    At = mat.Matrix(D).transpose()

    i = 0
    while i < 10:
        i += 1

        Z = W.product(At).mat()[0]
        delta = [sigmoid(z) - y for y, z in zip(ground_truth, Z)]
        delta = mat.Matrix([delta]).transpose()

        Hinv = mat.Matrix(hessian(W, D)).inverse()

        W = W.add(Hinv.product(At.product(delta)).mult(-1 * lr).transpose())

    return W

def regression(D, ground_truth, init_W, method):

    if method == 'Gradient descent':
        W = gradient_descent(init_W, D, ground_truth, 1)
        subplot = 132

    elif method == 'Newton\'s method':
        W = newton_method(init_W, D, ground_truth, 1)
        subplot = 133

    result = predict(W, mat.Matrix(D))
    confusion_matrix = get_confusion_matrix(result, ground_truth)

    print_result(method, W, confusion_matrix)
    plot_data_points(D, result, subplot)


N = 50
mxs = [1, 10]
vxs = [2, 2]
mys = [1, 10]
vys = [2, 2]

D, ground_truth = generate_ground_truth(N, mxs, vxs, mys, vys)
plot_data_points(D, ground_truth, 131)

init_W = random.gaussian(size=3)
regression(D, ground_truth, init_W, 'Gradient descent')
regression(D, ground_truth, init_W, 'Newton\'s method')

plt.savefig('test.png')

