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
from utils import load_data, order_extend_data, diff_perm_per_classifier, two_pixel_perm, two_pixel_perm_sliding
import numpy as np

conf = sys.argv[-1]
_type = sys.argv[-2]
with open(conf) as config_file:
    config = json.load(config_file)

model_dir = config['model_dir']
nb_labels = config['num_labels']
path = config['permutation']
st_lab = config['start_label']
#np.random.seed(st_lab)
# lab_perm = np.random.permutation(np.load('2_label_permutation.npy')[:nb_labels].T)#[st_lab:st_lab+nb_labels].T)
lab_perm = np.load('2_label_permutation.npy')[st_lab:st_lab+nb_labels].T

# Setting up training parameters
tf.set_random_seed(config['random_seed'])
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
loss_func = config['loss_func']
batch_size = config['training_batch_size']
nb_channal = int(path.split('_')[1].split('.')[1])

if _type == 'diff':
  imgs, labels, input_shape, model_dir = diff_perm_per_classifier(st_lab, nb_channal, model_dir)
elif _type == 'two':
  imgs, labels, input_shape, model_dir = two_pixel_perm_img(nb_channal, model_dir)
elif _type == 'slide':
  imgs, labels, input_shape, model_dir = two_pixel_perm_sliding(nb_channal, model_dir)
elif _type == 'normal':
  imgs, labels, input_shape = load_data(config['permutation'], config['num_labels'])
  

print(input_shape)
if loss_func != 'xent':
  labels = np.array([lab_perm[i] for i in labels]).astype(np.float32)
  labels[labels==0]=-1
  if loss_func == 'balance':
      labels[labels==0]=-1
model = keras.Sequential([keras.layers.Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape[1:]),
	keras.layers.MaxPooling2D(pool_size=(2,2)),
	keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'),
	keras.layers.MaxPooling2D(pool_size=(2,2)),
	keras.layers.Flatten(),
	keras.layers.Dense(1024, activation='relu'),
	keras.layers.Dense(nb_labels)
	])

def custom_loss(y_true, y_pred):
	if loss_func == 'bce':
		loss = keras.losses.BinaryCrossentropy()
		return loss(y_true, tf.nn.sigmoid(y_pred))
	elif loss_func == 'xent':
		loss = keras.losses.SparseCategoricalCrossentropy()
		return loss(y_true, keras.activations.softmax(y_pred))
	elif loss_func == 'balance':
#		y_true[y_true==0]=-1
		return -1*np.sum(y_true*(y_pred-.5))
model.compile(loss=custom_loss, optimizer=keras.optimizers.Adam(1e-3))

x_train, y_train = imgs[:60000], labels[:60000]
x_test, y_test = imgs[60000:], labels[60000:]

epochs = max_num_training_steps * batch_size / len(x_train)

#<<<<<<< HEAD
#model.fit(x_train, y_train, batch_size=batch_size, epochs=int(epochs), verbose=2, validation_data=(x_test,y_test))
#=======
chkpt_cb = tf.keras.callbacks.ModelCheckpoint(model_dir+'.h5',
                                              monitor='val_loss',
                                              save_best_only=True,
                                              mode='min')

model.fit(x_train, y_train, batch_size=batch_size, epochs=int(epochs), verbose=2, validation_data=(x_test,y_test), callbacks=[chkpt_cb])
#>>>>>>> refs/remotes/origin/master

#model.save(model_dir+'.h5')

