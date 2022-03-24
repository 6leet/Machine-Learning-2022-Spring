from asyncore import read
from cProfile import label
import yaml
import sys
import matplotlib.pyplot as plt

def read_yaml(filename):
    with open(filename, 'rb') as file:
        data = yaml.safe_load(file)
    return data['training-image'], data['training-label'], data['testing-image'], data['training-label']

def raw2image(raw, rows, cols):
    image = [[raw[i + j * rows] for i in range(cols)] for j in range(rows)]
    # plt.imshow(image, cmap='gray')
    # plt.show()
    return image

def preprocess(_image_file, _label_file): # non-intel: high endian, intel: low endian
    image_file = open(_image_file, 'rb')
    image_file.read(4)
    number_of_images = int.from_bytes(image_file.read(4), byteorder='big')
    rows = int.from_bytes(image_file.read(4), byteorder='big')
    cols = int.from_bytes(image_file.read(4), byteorder='big')
    
    image_byte = rows * cols
    raw = image_file.read(image_byte)
    images = []
    for _ in range(number_of_images - 1):
        raw = image_file.read(image_byte)
        image = raw2image(raw, rows, cols)
        images.append(image)

    label_file = open(_label_file, 'rb')
    label_file.read(4)
    number_of_labels = int.from_bytes(label_file.read(4), byteorder='big')
    print(number_of_labels)
    for _ in range(number_of_labels):
        label = int.from_bytes(label_file.read(1))
        print(label)
    


# example: python3 hw2.py 0 hw2.yaml
# 
if (len(sys.argv) < 3):
    print('Usage: python3 hw2.py {0|1} <yaml file>')
    exit()
train_image, train_label, test_image, test_label = read_yaml(sys.argv[2])
preprocess(train_image, train_label)