from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import sys
from utils import window_perm_sliding_img

def get_data(_type):
    label = np.load('data/mnist_labels.npy')
    if _type == 'origin':
        data = np.load('data/mnist_data.npy').astype(np.float32) / 255.
    elif _type == 'encode':
        data = np.load('data/mnist_data.npy').transpose((0,2,3,1))
        data = window_perm_sliding_img(32, data, 0, 1)
    return data.reshape(data.shape[0],-1), label

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    plt.title(title)
    return fig


def main(_type):
    data, label = get_data(_type)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0, method='barnes_hut')
#    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the MNIST')
#                         % (time() - t0))
    fig.savefig('pics/32_{}.png'.format(_type))


if __name__ == '__main__':
    _type = sys.argv[-1]
    main(_type)
