from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

from utils import load_data, extend_data
import numpy as np

conf = sys.argv[-1]
dataset = sys.argv[-2]
# Global constants
with open(conf) as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']
nb_labels = config['num_labels']
model_dir = config['model_dir']
st_lab = config['start_label']
rep = np.load('2_label_permutation.npy')[st_lab:st_lab+nb_labels].T

#if dataset == 'origin.npy':
# imgs, labels, input_shape = load_data(config['permutation'], config['num_labels'])
labels = np.load('data/mnist_labels.npy')
imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1))
permut = np.load(config['permutation'])
labels = np.array([rep[i] for i in labels]).astype(np.float32)
x_train, y_train = imgs[:60000], labels[:60000]
x_test, y_test = imgs[60000:], labels[60000:]

if len(x_test.shape) == 3:
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_test.shape, len(x_test))

def custom_loss():
  def loss(y_true, y_pred):
    if config['loss_func'] == 'bce':
      _loss = keras.losses.BinaryCrossentropy()
      return _loss(y_true, tf.nn.sigmoid(y_pred))
    elif config['loss_func'] == 'xent':
      _loss = keras.losses.SparseCategoricalCrossentropy()
      return _loss(y_true, tf.nn.softmax(y_pred))
  return loss

model = keras.models.load_model(model_dir+'.h5', custom_objects={ 'custom_loss': custom_loss(), 'loss':custom_loss() }, compile=False)

tot_advs_acc = np.zeros(len(y_test))
tot_amt = 0
while True:
  if np.mean(tot_advs_acc) == 1.: 
    print(tot_amt, 'totally attacked succeed!')
    break
  elif tot_amt == 1e5:
    print(tot_amt, 'total adversarial acc:', tot_advs_acc)
  else:
    tot_amt += 1
    noise = np.random.randint(0, int(config['epsilon']*255, x_test.shape))
    samples = np.array([[[permut[d[0]] for d in c] for c in b] for b in (x_test+noise).astype(np.int)])
    x_test = samples.astype(np.float32) / 255.
    output = model.predict(x_test, batch_size=eval_batch_size)
    nat_labels = np.zeros(output.shape).astype(np.float32)
    nat_labels[output>=0.5] = 1.
    nat_dists = np.sum(np.absolute(nat_labels-y_test), axis=-1)
    tot_advs_acc[nat_dists == 0] = 1.

    print('natural: {:.2f}%'.format(100 * np.mean(nat_dists == 0)))
    # np.save('preds/pred_{}_{}'.format(model_dir.split('/')[1], dataset.split('/')[-1]), output)


