import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def convert(filename_prefix):
    iter = 0
    imgs = []
    fig = plt.figure()
    while True:
        filename = '{}_table_{}.txt'.format(filename_prefix, iter)
        try:
            file = open(filename, 'r')
        except:
            break
        lines = file.readlines()

        img = [[0] * 100 for _ in range(100)]
        for p, line in zip(range(len(lines)), lines):
            x = p // 100
            y = p % 100
            img[x][y] = int(line)

        os.remove(filename)

        img = plt.imshow(img, animated=True)
        imgs.append([img])
        iter += 1

    ani = animation.ArtistAnimation(fig, imgs, interval=200, repeat_delay=1000)
    ani.save('{}.gif'.format(filename_prefix), writer='pillow')

convert(sys.argv[1])