import numpy as np
from numpy import isneginf
import yaml
import sys
from matplotlib import pyplot as plt
import random_data_generator as random

def read_yaml(filename):
    with open(filename, 'rb') as file:
        data = yaml.safe_load(file)
    return data['training-image'], data['training-label'], data['testing-image'], data['testing-label']

def raw2image(raw, rows, cols):
    image = [[round(raw[i + j * rows] / 255) for i in range(cols)] for j in range(rows)]
    return image

def preprocess(_image_file, _label_file): ## non-intel: high endian, intel: low endian
    image_file = open(_image_file, 'rb')
    image_file.read(4)
    number_of_images = int.from_bytes(image_file.read(4), byteorder='big')
    rows = int.from_bytes(image_file.read(4), byteorder='big')
    cols = int.from_bytes(image_file.read(4), byteorder='big')

    ## process image
    print('processing', _image_file, 'with', number_of_images, 'images...')
    image_byte = rows * cols
    images = []
    for _ in range(number_of_images):
        raw = image_file.read(image_byte)
        image = raw2image(raw, rows, cols)
        images.append(image)

    ## process label
    label_file = open(_label_file, 'rb')
    label_file.read(4)
    number_of_labels = int.from_bytes(label_file.read(4), byteorder='big')

    print('processing', _label_file, 'with', number_of_labels, 'labels...')
    raw = label_file.read(number_of_labels)
    labels = [r for r in raw]

    return images, labels

def difference(mu, new_mu):
    ans = 0
    for i in range(10):
        for j in range(784):
            ans += abs(mu[i,j] - new_mu[i,j])
    
    return ans

def display_num(mean):
    for i in range(10):
        print("\nclass: ", i)
        for j in range(28):
            for k in range(28):
                if (mean[i][j*28 + k] > 0.5):
                    print("1", end=" ")
                else:
                    print("0", end=" ")
            print("")
        print("")

def get_X(train_images, train_labels):
    h = len(train_images[0])
    w = len(train_images[0][0])
    X = [[[] for _ in range(h)] for _ in range(w)]
    for image, label in zip(train_images, train_labels):
        for i in range(h):
            for j in range(w):
                X[i][j].append(image[i][j])

    return X

def get_init(images, labels):
    h = len(images[0])
    w = len(images[0][0])
    
    _, cnt = np.unique(train_labels, return_counts=True)
    lamb = cnt / sum(cnt)

    Ps = np.zeros((10, h, w))
    for image, n in zip(images, labels):
        Ps[n] += np.array(image)

    for n in range(len(Ps)):
        Ps[n] /= cnt[n]
    
    return lamb, Ps

def em_algorithm(images, labels):
    lamb, Ps = get_init(images, labels)

    h = len(images[0])
    w = len(images[0][0])

    N = len(labels)
    cnt = 0
    labels = []
    while cnt < 10:
        cnt += 1

        # E
        W = np.zeros((N, 10))

        for k in range(N):
            outcome_prob = np.log(lamb)
            for n in range(10):
                P = np.log(Ps[n])
                P[isneginf(P)]=-15
                image = np.log(images[k])
                image[isneginf(image)]=-15

                _P = -P
                _image = np.ones((h, w)) - image
                outcome_prob[n] += sum(P * image) + sum(_P * _image)

            outcome_prob = np.exp(outcome_prob)
            W[k] = outcome_prob / sum(outcome_prob)
            max_label = np.argmax(W[k])

            labels.append(max_label)

        # M
        sum_W = sum(W.T)
        lamb = sum_W / N

        _Ps = Ps.copy()
        Ps = np.zeros((10, h, w))
        for n in range(10):
            for k in range(N):
                Ps[n] += W[k][n] * train_images[k]
            Ps[n] /= sum_W[n]

        diff = float(difference(Ps, _Ps))    
        display_num(Ps)

        print("----------------#", cnt, "iteration, differance: ", diff, "----------------")


if (len(sys.argv) < 2):
    print('Usage: python3 hw2.py <yaml file>')
    exit()
train_image_file, train_label_file, test_image_file, test_label_file = read_yaml(sys.argv[1])
train_images, train_labels = preprocess(train_image_file, train_label_file)
test_images, test_labels = preprocess(test_image_file, test_label_file)
print(len(test_images), len(test_labels))
_em_algorithm(train_images, train_labels)
# train_images = train_images[0:1000]
# train_labels = train_labels[0:1000]
# test_images = test_images[0:10]
# test_labels = test_labels[0:10]