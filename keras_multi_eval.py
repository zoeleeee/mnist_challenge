#python keras_multi_eval.py 1 HASH origin.npy configs/
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import sys
import time

from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

#from utils import *
import numpy as np
print(sys.argv[-1], sys.argv[-2], sys.argv[-3], sys.argv[-4])
conf = sys.argv[-1]
dataset = sys.argv[-2]
_type = sys.argv[-3]
input_bytes = eval(sys.argv[-4])
# Global constants
with open(conf) as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']
nb_labels = config['num_labels']
model_dir = config['model_dir']
st_lab = config['start_label']
rep = np.load('data/2_label_permutation.npy')[st_lab:st_lab+nb_labels].T
nb_channal = int(config['permutation'].split('_')[1].split('.')[1])
loss_func = config['loss_func']
#if dataset == 'origin.npy':
# np.random.seed(st_lab)
# perm = []
# for i in range(nb_channal):
#   perm.append(np.random.permutation(np.arange(256)))
# perm = np.array(perm).transpose((1,0))
# imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1))
# imgs = order_extend_data(perm, imgs)
# labels = np.load('data/mnist_labels.npy')
# input_shape = imgs.shape[1:]
print(model_dir)
if dataset != 'origin.npy':
  x_test = np.load(dataset)
  y_test = np.load(dataset[:-8]+'label.npy')
  y_test = np.array([rep[i] for i in y_test]).astype(np.float32)
  if dataset.endswith('show.npy'):
    if _type == 'normal':
      x_test = extend_data(config['permutation'], x_test)
    elif _type == 'two':
      x_test = two_pixel_perm_img(nb_channal, x_test)
    elif _type == 'slide':
      x_test = two_pixel_perm_sliding_img(nb_channal, x_test, st_lab)
    elif _type == 'slide4':
      x_test = four_pixel_perm_sliding_img_AES(nb_channal, x_test, st_lab, input_bytes)
    elif _type == 'window':
      x_test = window_perm_sliding_img_AES(nb_channal, x_test, st_lab, input_bytes)
    elif _type == 'diff':
      x_test = diff_perm_per_classifier_img(st_lab, nb_channal, x_test)
    elif _type == 'HASH':
      x_test = window_perm_sliding_img(nb_channal, x_test, st_lab, input_bytes)
      model_dir += '_HASH'+str(input_bytes) 
else:
  if _type == 'diff':
    imgs, labels, input_shape, model_dir = diff_perm_per_classifier(st_lab, nb_channal, model_dir)
  elif _type == 'two':
    imgs, labels, input_shape, model_dir = two_pixel_perm(nb_channal, model_dir)
  elif _type == 'slide':
    imgs, labels, input_shape, model_dir = two_pixel_perm_sliding(nb_channal, model_dir, st_lab)
  elif _type == 'normal':
    imgs, labels, input_shape = load_data(config['permutation'], config['num_labels'])
  elif _type == 'slide4':
    imgs, labels, input_shape, model_dir = four_pixel_perm_sliding_AES(nb_channal, model_dir, st_lab, input_bytes)
  elif _type == 'window':
    imgs, labels, input_shape, model_dir = window_perm_sliding_AES(nb_channal, model_dir, st_lab, input_bytes)
  elif _type == 'HASH':
    imgs, labels, input_shape, model_dir = window_perm_sliding(nb_channal, model_dir, st_lab, input_bytes)
  labels = np.array([rep[i] for i in labels]).astype(np.float32)
  #x_train, y_train = imgs[:60000], labels[:60000]
  x_test, y_test = imgs[60000:], labels[60000:]
print(model_dir)

if len(x_test.shape) == 3:
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_test.shape, len(x_test))
y_test = y_test[:len(x_test)]
def custom_loss():
  def loss(y_true, y_pred):
    if loss_func == 'bce':
      _loss = keras.losses.BinaryCrossentropy()
      return _loss(y_true, tf.nn.sigmoid(y_pred))
    elif loss_func == 'xent':
      _loss = keras.losses.SparseCategoricalCrossentropy()
      return _loss(y_true, tf.nn.softmax(y_pred))
    elif loss_func == 'balance':
      y_true[y_true==0]=-1
      return -1*np.sum(y_true*(y_pred-.5))
  return loss

print(model_dir)
model = keras.models.load_model(model_dir+'.h5', custom_objects={ 'custom_loss': custom_loss(), 'loss':custom_loss() }, compile=False)

output = model.predict(x_test, batch_size=eval_batch_size)
nat_labels = np.zeros(output.shape).astype(np.float32)
nat_labels[output>=0.5] = 1.
nat_dists = np.sum(np.absolute(nat_labels-y_test), axis=-1)
nat_acc = np.mean(nat_dists == 0)

print('natural: {:.2f}%'.format(100 * nat_acc))
np.save('preds/pred_{}_{}'.format(model_dir.split('/')[1], dataset.split('/')[-1]), output)


