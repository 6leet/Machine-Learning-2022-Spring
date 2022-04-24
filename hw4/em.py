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

def get_X(train_images, train_labels):
    h = len(train_images[0])
    w = len(train_images[0][0])
    X = [[[] for _ in range(h)] for _ in range(w)]
    for image, label in zip(train_images, train_labels):
        for i in range(h):
            for j in range(w):
                X[i][j].append(image[i][j])

    return X

def get_init(train_images, train_labels):
    Xs = get_X(train_images, train_labels)

    h = len(train_images[0])
    w = len(train_images[0][0])
    Ps = [[[] for _ in range(h)] for _ in range(w)]

    cnt = [0] * 10
    for n in train_labels:
        cnt[n] += 1
    lams = [[[cnt[n] / sum(cnt) for n in range(len(cnt))] for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            X = Xs[i][j]
            cnt_one = [0] * 10
            for x, n in zip(X, train_labels):
                cnt_one[n] += x
            
            Ps[i][j] = [cnt_one[n] / cnt[n] for n in range(len(cnt))]
    
    return Xs, lams, Ps

def _em_algorithm(train_images, train_labels):
    Xs, lams, Ps = get_init(train_images, train_labels)

    h = len(train_images[0])
    w = len(train_images[0][0])

    cnt = 0
    while cnt < 100:
        cnt += 1

        for i in range(h):
            for j in range(w):
                print(str(cnt) + ':', i, j)
                # E
                W = [[], []]
                for a in range(2):
                    s = sum([l * (1 - a + max(p, 1e-4)) for l, p in zip(lams[i][j], Ps[i][j])])
                    W[a] = [l * (1 - a + max(p, 1e-4)) / s for l, p in zip(lams[i][j], Ps[i][j])]
                # M
                for n in range(len(Ps[i][j])):
                    lams[i][j][n] = sum([W[x][n] for x in Xs[i][j]]) / len(Xs[i][j])
                    Ps[i][j][n] = W[1][n] * sum(Xs[i][j]) / (lams[i][j][n] * len(Xs[i][j]))

            
        print('Iteration', cnt)
        for n in range(10):
            image = []
            for i in range(h):
                row = []
                for j in range(w):
                    # row.append(Ps[i][j][n])
                    print(round(Ps[i][j][n]), end=' ')
                print()
            print()
            #     image.append(row)
            # plt.imshow(image, cmap='gray')
            # plt.show()

def em_algorithm(train_images, train_labels):
    Xs, lams, Ps = get_init(train_images, train_labels)

    N = len(train_labels)
    cnt = 0
    while cnt < 10:
        cnt += 1

        W = [[0] * 10 for _ in range(N)]


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