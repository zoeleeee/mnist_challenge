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
imgs, labels, input_shape = load_data(config['permutation'], config['num_labels'])
labels = np.array([rep[i] for i in labels]).astype(np.float32)
x_train, y_train = imgs[:60000], labels[:60000]
x_test, y_test = imgs[60000:], labels[60000:]
if dataset != 'origin.npy':
  x_test = np.load(dataset)
  if dataset.endswith('show.npy'):
    x_test = extend_data(config['permutation'], x_test)

if len(x_test.shape) == 3:
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_test.shape, len(x_test))

model = keras.models.load_model(model_dir+'.h5')

output = model.predict(x_test, batch_size=eval_batch_size)
nat_labels = np.zeros(output.shape).astype(np.float32)
nat_labels[output>=0.5] = 1.
nat_dists = np.sum(np.absolute(nat_labels-y_test), axis=-1)
nat_acc = np.mean(nat_dists == 0)

print('natural: {:.2f}%'.format(100 * nat_acc))
np.save('preds/pred_{}_{}'.format(model_dir.split('/')[1], dataset.split('/')[-1]), output)


