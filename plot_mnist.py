import matplotlib.pyplot as plt
import numpy as np
import sys

file = sys.argv[-1]
label = int(sys.argv[-2])
idx = file.split('_')[-2]
imgs = np.load(file)
for i, img in enumerate(imgs):
    plt.imshow(img)
    plt.savefig("mnist_pics/fig#{}_{}.png".format(idx, (label+i)%10), cmap='Greys')