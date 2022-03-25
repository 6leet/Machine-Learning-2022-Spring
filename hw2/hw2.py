from asyncore import read
from cProfile import label
import yaml
import sys
import matplotlib.pyplot as plt
import math

def read_yaml(filename):
    with open(filename, 'rb') as file:
        data = yaml.safe_load(file)
    return data['training-image'], data['training-label'], data['testing-image'], data['testing-label']

def raw2image(raw, rows, cols):
    image = [[raw[i + j * rows] for i in range(cols)] for j in range(rows)]
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
        if raw:
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

def learn_discrete(train_images, train_labels):
    print('training in discrete mode...')
    cols, rows = len(train_images[0]), len(train_images[0][0])
    bins = 32
    conp = [[[[0] * bins for _ in range(cols)] for _ in range(rows)] for _ in range(10)]
    p = [0] * 10
    
    for image, label in zip(train_images, train_labels):
        p[label] += 1
        for i in range(rows):
            for j in range(cols):
                ## for a number `label`, the frequency of pixel (i, j) having bit N (image[i][j])
                conp[label][i][j][int(image[i][j] / 8)] += 1
    return conp, p

def test_discrete(test_images, test_labels, conp, p):
    cols, rows = len(test_images[0]), len(test_images[0][0])

    p_sum = sum(p)
    p_28_28 = [pow(p[i], 28 * 28) for i in range(len(p))]

    posts = [0] * 10
    err = 0
    for image, label in zip(test_images, test_labels):
        print('Posterior (in log scale):')
        predict = -1
        max_post = -1
        for n in range(10):
            jp = 1
            for i in range(rows):
                for j in range(cols):
                    jp = jp * conp[n][i][j][int(image[i][j] / 8)]
            posts[n] = jp / p_28_28[n] * p[n] / p_sum
            if (posts[n] != 0):
                posts[n] = -1 / math.log(posts[n])

        posts_sum = sum(posts) if sum(posts) != 0 else 1
        for n in range(10):
            post = posts[n] / posts_sum
            print(str(n) + ':', post)
            if (post > max_post):
                max_post = post
                predict = n
        print('Prediction:', predict, 'Ans:', label)
        if predict != label:
            err += 1

    return err / len(test_images)

def learn_continuous(train_images, train_labels):
    print('training in continuous mode...')
    cols, rows = len(train_images[0]), len(train_images[0][0])
    scale_sum = [[[0] * cols for _ in range(rows)] for _ in range(10)]
    scale_list = [[[[] for _ in range(cols)] for _ in range(rows)] for _ in range(10)]
    p = [0] * 10

    for image, label in zip(train_images, train_labels):
        p[label] += 1
        for i in range(rows):
            for j in range(cols):
                scale_sum[label][i][j] += image[i][j]
                scale_list[label][i][j].append(image[i][j])

    mean = [[[scale_sum[n][i][j] / p[n] for j in range(cols)] for i in range(rows)] for n in range(10)]
    var = [[[sum([pow(mean[n][i][j] - s, 2) for s in scale_list[n][i][j]]) / p[n] for j in range(cols)] for i in range(rows)] for n in range(10)]

    return mean, var, p

def gaussian(mean, var, x):
    if var == 0: ## make approximation to dirac delta function (try)
        var = 500
    return 1 / math.sqrt(2 * math.pi * var) * math.exp(-pow(x - mean, 2) / (2 * var))

def test_continuous(test_images, test_labels, mean, var, p):
    cols, rows = len(test_images[0]), len(test_images[0][0])

    p_sum = sum(p)

    posts = [0] * 10
    err = 0
    for image, label in zip(test_images, test_labels):
        print('Posterior (in log scale):')
        predict = -1
        max_post = -1
        for n in range(10):
            jp = 0
            for i in range(rows):
                for j in range(cols):
                    # if image[i][j] != 0:
                    gauss = gaussian(mean[n][i][j], var[n][i][j], image[i][j])
                    if gauss == 0: ## make probability small enough, but not zero
                        gauss = 1 / 60000
                    gauss = math.log(gauss) # if gauss != 0 else 0
                    jp = jp + gauss

            posts[n] = jp + math.log(p[n] / p_sum)
            if (posts[n] != 0):
                posts[n] = -1 / posts[n]

        posts_sum = sum(posts) if sum(posts) != 0 else 1
        for n in range(10):
            post = posts[n] / posts_sum
            print(str(n) + ':', post)
            if (post > max_post):
                max_post = post
                predict = n
        print('Prediction:', predict, 'Ans:', label)
        if predict != label:
            err += 1

    return err / len(test_images)

def naive_bayes(train_images, train_labels, test_images, test_labels, mode):
    if mode == 0:
        conp, p = learn_discrete(train_images, train_labels)
        err = test_discrete(test_images, test_labels, conp, p)
        print('Error rate:', err)
    elif mode == 1:
        mean, var, p = learn_continuous(train_images, train_labels)
        # image = []
        # for i in range(28):
        #     row = []
        #     for j in range(28):
        #         print(mean[0][i][j], end=' ')
        #         if (mean[0][i][j] < 128):
        #             row.append(0)
        #         else:
        #             row.append(1)
        #     image.append(row)
        #     print()
        # plt.imsave('wtf', image, cmap='gray', format='png')
        err = test_continuous(test_images, test_labels, mean, var, p)
        print('Error rate', err)

## example: python3 hw2.py 0 hw2.yaml
if (len(sys.argv) < 3):
    print('Usage: python3 hw2.py {0|1} <yaml file>')
    exit()
train_image_file, train_label_file, test_image_file, test_label_file = read_yaml(sys.argv[2])
train_images, train_labels = preprocess(train_image_file, train_label_file)
test_images, test_labels = preprocess(test_image_file, test_label_file)
# train_images = train_images[0:12000]
# train_labels = train_labels[0:12000]
# test_images = test_images[0:1000]
# test_labels = test_labels[0:1000]
naive_bayes(train_images, train_labels, test_images, test_labels, int(sys.argv[1]))