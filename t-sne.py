import numpy as numpy
from utils import order_extend_data
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

color= ['goldenrod', 'yellowgreen', 'blue', 'orchid', 'tomato', 'orange', 'aquamarine', 'purple', 'black', 'tan']
def plot(x, lab, name):
	plt.cla()
	nb_lab = np.max(lab)+1
	for i in range(nb_lab):
		point = x[lab==i]
		plt.scatter(point[:,0], point[:,1], color=color[i])
	plt.savefig(name)

nb_channel = int(sys.argv[-1])
data = np.load('data/mnist_data.npy').transpose((0,2,3,1))[60000:]
label = np.load('data/mnist_labels.npy')[60000:]
X_embedded = TSNE(n_components=2).fit_transform(data)
plot(X_embedded, label, 'mnist_pics/origin.png')
for i in range(0,100,20):
	np.random.seed(i)
	perm = np.array([np.random.permutation(np.arange(256)) for j in range(nb_channel)]).transpose((1,0))
	imgs = order_extend_data(perm, data)
	X_embedded = TSNE(n_components=2).fit_transform(imgs)
	plot(X_embedded, label, 'mnist_pics/{}_perm.png'.format(i))
