import sys
from PIL import Image
import numpy as np

def convert(filename):
    img = np.asarray(Image.open(filename))
    text_filename = '{}.txt'.format(filename[:-4])
    with open(text_filename, 'w') as file:
        for row in img:
            for pixel in row:
                for color in pixel:
                    file.write(f'{color} ')
                file.write('\n')
    file.close()

convert(sys.argv[1])