"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
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

from multi_model import Model
from pgd_attack import LinfPGDAttack
from utils import load_data
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
rep = np.load('2_label_permutation.npy')[:nb_labels].T

#if dataset == 'origin.npy':
imgs, labels, input_shape = load_data(config['permutation'], config['num_labels'])
labels = np.array([rep[i] for i in labels]).astype(np.float32)
x_train, y_train = imgs[:60000], labels[:60000]
x_test, y_test = imgs[60000:], labels[60000:]
if dataset != 'origin.npy':
  x_test = np.load(dataset)
  if dataset.endswith('show.npy'):
    x_test = extend_data(config['permutation'], x_test)

# Set upd the data, hyperparameters, and the model
# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

# if eval_on_cpu:
#   with tf.device("/cpu:0"):
#     model = Model()
#     attack = LinfPGDAttack(model, 
#                            config['epsilon'],
#                            config['k'],
#                            config['a'],
#                            config['random_start'],
#                            config['loss_func'])
# else:
#   model = Model()
#   attack = LinfPGDAttack(model, 
#                          config['epsilon'],
#                          config['k'],
#                          config['a'],
#                          config['random_start'],
#                          config['loss_func'])
model = Model(input_shape[-1], config['num_labels'])
global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

# last_checkpoint_filename = ''
already_seen_state = False

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, filename)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    bce_score_nat = []
    bce_labels = []
    # total_xent_adv = 0.
    total_corr_nat = 0
    avg_nat_acc = 0
    # total_corr_adv = 0

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = x_test[bstart:bend, :]
      y_batch = y_test[bstart:bend]

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch}

      # x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      # dict_adv = {model.x_input: x_batch_adv,
      #             model.y_input: y_batch}

      cur_nat_score, cur_bce_nat = sess.run(
                                      [model.bce_score,model.bce_loss],
                                      feed_dict = dict_nat)
      # cur_corr_adv, cur_xent_adv = sess.run(
      #                                 [model.num_correct,model.xent],
      #                                 feed_dict = dict_adv)

      if len(bce_score_nat) == 0:
        bce_score_nat = np.array(cur_nat_score)
        bce_labels = np.array(y_batch)
      else:
        bce_score_nat = np.vstack((bce_score_nat, cur_nat_score))
        bce_labels = np.vstack((bce_labels, y_batch))

      cur_nat_score = np.array(cur_nat_score)
      nat_labels = np.zeros(cur_nat_score.shape).astype(np.float32)
      nat_labels[cur_nat_score>=0.5] = 1.
      nat_dists = np.sum(np.absolute(nat_labels-y_batch), axis=-1)
      nat_acc = np.sum(nat_dists == 0)
      # print(nat_dists)
      avg_nat_acc += nat_acc
      #bce_score_nat.append(cur_nat_score)
      # total_xent_adv += cur_xent_adv
      total_corr_nat += cur_bce_nat
      # total_corr_adv += cur_corr_adv

    # avg_xent_nat = total_xent_nat / num_eval_examples
    # avg_xent_adv = total_xent_adv / num_eval_examples
    avg_bce_nat = total_corr_nat / num_eval_examples
    avg_nat_acc /= num_eval_examples
    # acc_adv = total_corr_adv / num_eval_examples

    # summary = tf.Summary(value=[
    #       # tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
    #       # tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
    #       tf.Summary.Value(tag='xent nat', simple_value= acc_bce_nat),
    #       # tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
    #       # tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
    #       # tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
    # summary_writer.add_summary(summary, global_step.eval(sess))

    print('natural: {:.2f}%'.format(100 * avg_nat_acc))
    # # print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_bce_nat))
    np.save('preds/pred_{}_{}'.format(model_dir.split('/')[1], dataset.split('/')[-1]), bce_score_nat)
    np.save('preds/labels_{}_{}'.format(model_dir.split('/')[1], dataset.split('/')[-1]), bce_labels)
    # print('avg adv loss: {:.4f}'.format(avg_xent_adv))

# Infinite eval loop
# while True:
cur_checkpoint = tf.train.latest_checkpoint(model_dir)
evaluate_checkpoint(cur_checkpoint)
