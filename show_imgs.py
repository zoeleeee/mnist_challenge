import matplotlib.pyplot as plt
import numpy as np
import sys


def show_imgs(path):
	org_imgs = np.load('data/mnist_data.npy')
	org_imgs = np.load('data/mnist_labels.npy')
	preds = np.load('pred.npy')
	if not os.path.exists(path[:-4]):
		os.makedirs(path[:-4])
	imgs = np.load(path)
	imgs_shape = imgs.shape
	if len(imgs_shape) > 4:
		if imgs_shape
	for img in imgs:
		plt.imshow()
		plt.savefig(os.path.join(path[:-4], ))


if __name__ == '__main__':
	path = sys.argv[-1]
	show_imgs(path)