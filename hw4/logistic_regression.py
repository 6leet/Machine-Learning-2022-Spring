import random_data_generator as random
from matplotlib import pyplot as plt
import matrix as mat
import math

def plot_data_points(D, color):
    for d in D:
        plt.plot(d[0], d[1], color + '.')

def plot_all(Ds, colors):
    for D, color in zip(Ds, colors):
        plot_data_points(D, color)

def plot_ground(Ds, colors):
    plt.subplot(131)
    plot_all(Ds, colors)

def plot_gradient(Ds, colors):
    plt.subplot(132)
    plot_all(Ds, colors)

def plot_newton(Ds, colors):
    plt.subplot(133)
    plot_all(Ds, colors)

def print_confusion(confusion_matrix):
    print('Confusion Matrix:')
    s = 'predict cluster 1 predict cluster 2'
    s1 = 'Is cluster 1'
    s2 = 'Is cluster 2'
    print(s.rjust(len(s) + len(s1)))
    print(s1 + str(confusion_matrix[0][0]).center(len(s) // 2) + str(confusion_matrix[0][1]).center(len(s) // 2))
    print(s2 + str(confusion_matrix[1][0]).center(len(s) // 2) + str(confusion_matrix[1][1]).center(len(s) // 2))

def generate_data_points(N, mean_x, var_x, mean_y, var_y):
    X = random.gaussian(mean_x, var_x, N)
    Y = random.gaussian(mean_y, var_y, N)
    D = [[x, y] for x, y in zip(X, Y)]
    return D

def generate_ground_truth(N, mxs, vxs, mys, vys):
    Ds = []
    for i in range(len(mxs)):
        Ds.append(generate_data_points(N, mxs[i], vxs[i], mys[i], vys[i]))
    return Ds[0], Ds[1]

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def get_train_and_label(D1, D2):
    for i in range(len(D1)):
        D1[i].append(1) 
    for i in range(len(D2)):
        D2[i].append(1)

    ground_truth = [1] * len(D1)
    ground_truth.extend([0] * len(D2))

    D1.extend(D2)

    return D1, ground_truth

def gradient_descent(D1, D2, lr):
    D, ground_truth = get_train_and_label(D1, D2)
    W = mat.Matrix([random.gaussian(size=3)])

    error = 0
    last_error = 1

    while abs(error - last_error) > 1e-7 * len(D): # convergence rule?
        last_error = error
        error = 0

        delta = mat.Matrix([[0, 0, 0]])
        for y, d in zip(ground_truth, D):
            x = mat.Matrix([d])
            z = W.product(x.transpose()).access(0, 0)
            error += abs(y - sigmoid(z))
            delta = delta.add(x.mult(-(y - sigmoid(z))))

        W = W.add(delta.mult(-1 * lr))

    D1 = []
    D2 = []
    confusion_matrix = [[0, 0], [0, 0]]
    for y, d in zip(ground_truth, D):
        x = mat.Matrix([d])
        z = W.product(x.transpose()).access(0, 0)
        if (sigmoid(z) > 0.5):
            D1.append(d[:2])
            if (y == 1):
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][0] += 1
        else:
            D2.append(d[:2])
            if (y == 1):
                confusion_matrix[0][1] += 1
            else:
                confusion_matrix[1][1] += 1
    print('Gradient descent:\n')
    print('W:')
    W.transpose().show()

    print()
    print_confusion(confusion_matrix)

    print()
    print('Sensitivity (Successfully predict cluster 1):', confusion_matrix[0][0] / sum(confusion_matrix[0]))
    print('Specificity (Successfully predict cluster 2):', confusion_matrix[1][1] / sum(confusion_matrix[1]))

    plot_gradient([D1, D2], ['r', 'b'])

# def netwon_method()


N = 50
mxs = [1, 3]
vxs = [2, 4]
mys = [1, 3]
vys = [2, 4]

D1, D2 = generate_ground_truth(N, mxs, vxs, mys, vys)
plot_ground([D1, D2], ['r', 'b'])
gradient_descent(D1, D2, 2)
plt.savefig('test.png')

