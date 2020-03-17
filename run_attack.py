"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

from utils import load_data, extend_data

def run_attack(checkpoint, x_adv, config):#epsilon, permutation_path, nb_labels):
  epsilon = config['epsilon']
  imgs, labs, input_shape = load_data(config['permutation'])
  x_test, y_test = imgs[60000:], labs[60000:]
  y_adv = np.load('advs_targeted_labels.npy')
  # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
  if config['loss_func'] == 'bce':
    from multi_model import Model
  elif config['loss_func'] == 'xent':
    from  model import Model
  model = Model(input_shape[-1], config['num_labels'])

  saver = tf.train.Saver()

  num_eval_examples = 10000
  eval_batch_size = 50

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0

  x_nat = x_test
  l_inf = np.amax(np.abs(x_nat - x_adv))
  
  if l_inf > epsilon + 0.0001:
    print('maximum perturbation found: {}'.format(l_inf))
    print('maximum perturbation allowed: {}'.format(epsilon))
    return

  y_pred = [] # label accumulator

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)
    amt, cor = 0, 0
    # Iterate over the samples batch-by-batch
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = x_adv[bstart:bend, :]
      y_batch = y_adv[bstart:bend]
      x_nat_batch = x_test[bstart:bend, :]
      y_nat_batch = y_test[bstart:bend]

      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch}
      dict_nat = {model.x_input: x_nat_batch,
                  model.y_input: y_batch}
      y_adv_pred_batch = np.array(sess.run([model.y_pred],
                                        feed_dict=dict_adv))[0]
      y_nat_pred_batch = np.array(sess.run([model.y_pred],
                                        feed_dict=dict_nat))[0]
      # print(y_batch.shape, y_adv_pred_batch.shape, y_nat_pred_batch.shape)

      # amt += np.sum(y_nat_pred_batch == y_batch)
      # cor += np.sum(y_adv_pred_batch[y_nat_pred_batch==y_batch] == y_batch[y_nat_pred_batch==y_batch])

      # total_corr += cur_corr
      y_pred.append([y_nat_pred_batch, y_adv_pred_batch])

  # accuracy = cor / amt

  # print('Accuracy: {} / {} = {:.2f}%'.format(cor, amt, 100.0 * accuracy))
  y_pred = np.concatenate(y_pred, axis=0).transpose((1,0))
  idxs = np.arange(len(y_pred))[y_test != y_adv]
  idxs = idxs[y_pred[0][idxs] == y_test[idxs]]
  cor = np.sum(y_pred[idxs] == y_test[idxs])
  adv_cor = np.sum(y_pred[idxs] == y_adv[idxs])
  print('Accuracy: {} / {} = {:.2f}%'.format(cor, len(idxs), cor/len(idxs)))
  print('Adversarial Accuracy: {} / {} = {:.2f}%'.format(adv_cor, len(idxs), adv_cor/len(idxs)))

  # np.save('pred.npy', y_pred)
  # print('Output saved at pred.npy')

if __name__ == '__main__':
  import json
  import sys

  conf = sys.argv[-1]

  with open(conf) as config_file:
    config = json.load(config_file)

  model_dir = config['model_dir']

  checkpoint = tf.train.latest_checkpoint(model_dir)
  x_adv = np.load(config['store_adv_path'][:-10]+'show.npy')
  x_adv = extend_data(config['permutation'], x_adv)

  if checkpoint is None:
    print('No checkpoint found')
  elif len(x_adv.shape) != 4:
    print('Invalid shape: expected (10000,28,28,-1), found {}'.format(x_adv.shape))
  elif np.amax(x_adv) > 1.0001 or \
       np.amin(x_adv) < -0.0001 or \
       np.isnan(np.amax(x_adv)):
    print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
                                                              np.amin(x_adv),
                                                              np.amax(x_adv)))
  else:
    run_attack(checkpoint, x_adv, config)
