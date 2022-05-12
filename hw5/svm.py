from libsvm.svmutil import *
import csv
import numpy as np

# libsvm format: <label> <index1>:<value1> <index2>:<value2>

def preprocess(labelfile, datafile):
    labels = []
    with open(labelfile, newline='') as file:
        reader = csv.reader(file)
        for line in reader:
            labels.append(int(line[0]))

    keys = list(range(1, 28 * 28 + 1))
    datas = []
    with open(datafile, newline='') as file:
        reader = csv.reader(file)
        for line in reader:
            line = [float(x) for x in line]
            datas.append(dict(zip(keys, line)))

    return labels, datas

def grid_search(X, Y, kernel):
    kernel_method = {'linear': 0, 'polynomial': 1, 'rbf': 2}
    cost = [0.001, 0.01, 0.1, 1, 10, 100]
    degree = [2, 3, 4, 5, 6]
    gamma = [0.001, 0.01, 0.1, 1, 10, 100]
    coef0 = [1, 2, 3, 4, 5]
    prob = svm_problem(Y, X)
    if kernel == 'linear':
        opt_acc = -1
        opt_c = -1
        for c in cost:
            print('--------')
            print('cost:', c)
            param_cmd = '-t {} -c {} -v 5 -q'.format(kernel_method[kernel], c)
            param = svm_parameter(param_cmd)
            acc = svm_train(prob, param)
            if acc > opt_acc:
                opt_acc = acc
                opt_c = c
    elif kernel == 'polynomial':
        opt_acc = -1
        opt_c = -1
        opt_d = -1
        opt_r = -1
        for c in cost:
            for d in degree:
                for r in coef0:
                    print('--------')
                    print('cost:', c, 'degree:', d, 'coef0:', r)
                    param_cmd = '-t {} -c {} -d {} -r {} -v 5 -q'.format(kernel_method[kernel], c, d, r)
                    param = svm_parameter(param_cmd)
                    acc = svm_train(prob, param)
                    if acc > opt_acc:
                        opt_acc = acc
                        opt_c = c
                        opt_d = d
                        opt_r = r
        print('--------')
        print(opt_acc, 'cost =', opt_c, 'degree =', opt_d, 'coef0 =', opt_r)
    elif kernel == 'rbf':
        opt_acc = -1
        opt_c = -1
        opt_g = -1
        for c in cost:
            for g in gamma:
                print('--------')
                print('cost:', c, 'gamma:', g)
                param_cmd = '-t {} -c {} -g {} -v 5 -q'.format(kernel_method[kernel], c, g)
                param = svm_parameter(param_cmd)
                acc = svm_train(prob, param)
                if acc > opt_acc:
                    opt_acc = acc
                    opt_c = c
                    opt_g = g

def svm(X, Y, test_X, test_Y, kernel):

    kernel_method = {'linear': 0, 'polynomial': 1, 'rbf': 2}
    param_cmd = '-t {} -q'.format(kernel_method[kernel])
    
    prob = svm_problem(Y, X)
    param = svm_parameter(param_cmd)
    model = svm_train(prob, param)

    p_label, p_acc, p_val = svm_predict(test_Y, test_X, model)

def linear_kernel(X1, X2):
    return X1.dot(X2.T)

def rbf_kernel(X1, X2, gamma):
    dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-dist * gamma)

def dict2list(X):
    X_list = []
    for x in X:
        X_list.append(list(x.values()))

    return np.array(X_list)

def list2dict(X):
    X_dict = []
    for i, x in zip(range(len(X)), X):
        x = list(x)
        x.insert(0, i + 1)
        X_dict.append(dict(zip(list(range(len(x))), x)))

    return X_dict

def linear_rbf_svm(X, Y, test_X, test_Y):
    X = dict2list(X)
    test_X = dict2list(test_X)

    X_kernel = list2dict(list(linear_kernel(X, X) + rbf_kernel(X, X, 1 / (28 * 28))))
    test_X_kernel = list2dict(list(linear_kernel(X, test_X).T + rbf_kernel(X, test_X, 1 / (28 * 28)).T))

    param_cmd = '-t 4 -q'

    prob = svm_problem(Y, X_kernel)
    param = svm_parameter(param_cmd)
    model = svm_train(prob, param)
    
    p_label, p_acc, p_val = svm_predict(test_Y, test_X_kernel, model)
    

train_labels, train_datas = preprocess('ML_HW05/data/Y_train.csv', 'ML_HW05/data/X_train.csv')
test_labels, test_datas = preprocess('ML_HW05/data/Y_test.csv', 'ML_HW05/data/X_test.csv')

kernel = ['linear', 'polynomial', 'rbf']
for k in kernel:
    svm(train_datas, train_labels, test_datas, test_labels, k)
    grid_search(train_datas, train_labels, k)

linear_rbf_svm(train_datas, train_labels, test_datas, test_labels)