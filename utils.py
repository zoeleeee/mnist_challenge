import keras
import numpy as np
import os

def permutate_labels(labels, path='2_label_permutation.npy'):
	order = np.load(path)
	labs = [order[i] for i in labels]
	return np.array(labs)

def load_data(path, nb_labels=-1, one_hot=False):
	labels = np.load('data/mnist_labels.npy').astype(np.int)
	if one_hot:
		labels = keras.utils.to_categorical(labels, nb_labels)
	#labs = np.load('mnist_labels.npy')
	#label_permutation = np.load('2_label_permutation.npy')[:int(nb_labels)].T
	#labels = np.array([label_permutation[i] for i in labs])
	order = np.load(path)#'256_65536_permutation.npy'
	if os.path.exists('data/{}_mnist_data.npy'.format(path.split('_')[1])):
		imgs = np.load('data/{}_mnist_data.npy'.format(path.split('_')[1]))
		input_shape = (28, 28, int(path.split('_')[1].split('.')[-1]))
	# if eval(path.split('/')[-1].split('_')[1]) == 65536:
	# 	input_shape = (28, 28, 1)
	# 	imgs = np.load('65536_mnist_data.npy')
	# elif eval(path.split('/')[-1].split('_')[1]) == 256.2:
	# 	imgs = np.load('256_2_mnist_data.npy')
	# 	input_shape = (28, 28, 2)
	elif len(order.shape) > 1:
		input_shape = (28, 28, int(path.split('_')[1].split('.')[-1]))
		imgs = np.transpose(np.load('data/mnist_data.npy').astype(np.int), (1,0,2,3))[0]
		#imgs = np.clip(np.transpose(np.load('mnist_data.npy').astype(np.float32)+1, (1,0,2,3))[0], 0, 255).astype(np.int)

		tmp = np.array([copy.deepcopy(imgs) for i in np.arange(int(path.split('_')[1].split('.')[-1]))])
		samples = np.array([[[[order[d][i] for d in c] for c in b] for b in tmp[i]] for i in np.arange(tmp.shape[0])])
		imgs = np.transpose(samples, (1,2,3,0)).astype(np.float32) / (int(path.split('_')[1].split('.')[0])-1)
		np.save('data/{}_mnist_data.npy'.format(path.split('_')[1]),imgs)
	elif len(order.shape) == 1:
		input_shape = (28, 28, 1)
		imgs = np.transpose(np.load('data/mnist_data.npy'), (0,2,3,1)) 
		samples = np.array([[[[order[a] for a in b] for b in c] for c in d] for d in imgs])
		imgs = samples.astype(np.float32) / (int(path.split('_')[1].split('.')[0])-1)
		np.save('data/{}_mnist_data.npy'.format(path.split('_')[1]), imgs)
	return imgs, labels, input_shape
