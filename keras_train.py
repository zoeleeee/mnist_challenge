from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
#from mnist import get_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys
import os
import json
from utils import load_data
import numpy as np

conf = sys.argv[-1]
with open(conf) as config_file:
    config = json.load(config_file)

model_dir = config['model_dir']
nb_labels = config['num_labels']
path = config['permutation']
st_lab = config['start_label']
lab_perm = np.load('2_label_permutation.npy')[st_lab:st_lab+nb_labels].T

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

imgs, labels, input_shape = load_data(path, nb_labels)
if config['loss_func'] == 'bce':
  labels = np.array([lab_perm[i] for i in labels]).astype(np.float32)

model = keras.Sequential([keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(28,28,input_shape[-1])),
	keras.layers.MaxPooling2D(pool_size=(2,2)),
	keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu'),
	keras.layers.MaxPooling2D(pool_size=(2,2)),
	keras.layers.Flatten(),
	keras.layers.Dense(1024, activation='relu'),
	keras.layers.Dense(nb_labels)
	])

def custom_loss(y_true, y_pred):
	if config['loss_func'] == 'bce':
		loss = keras.losses.BinaryCrossentropy(y_true, y_pred)
	elif config['loss_func'] == 'xent':
		loss = keras.losses.SparseCategoricalCrossentropy(y_true, keras.activations.softmax(y_pred))
	return loss
model.compile(loss=custom_loss, optimizer=keras.optimizers.Adam(1e-3))

x_train, y_train = imgs[:60000], labels[:60000]
x_test, y_test = imgs[60000:], labels[60000:]

epochs = max_num_training_steps * batch_size / len(x_train)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test,y_test))

model.save(model_dir+'.h5')

