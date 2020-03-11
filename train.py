"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack

import sys
from utils import load_data

conf = sys.argv[-1]
with open(conf) as config_file:
    config = json.load(config_file)

nb_labels = config['num_labels']
path = config['permutation']
lab_perm = np.load('2_label_permutation.npy')[:nb_labels].T

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
# imgs, labels, input_shape = load_data(path, nb_labels)
imgs, labs, input_shape = load_data(path, nb_labels)
labels = np.array([lab_perm[i] for i in labs]).astype(np.float32)
print(labels.shape)
x_train, y_train = imgs[:60000], labels[:60000]
x_test, y_test = imgs[60000:], labels[60000:]
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(input_shape[-1], nb_labels)

# Setting up the optimizer
# train_step = tf.train.AdamOptimizer(1e-3).minimize(model.xent, global_step=global_step)
train_step = tf.train.AdamOptimizer(1e-3).minimize(model.bce_loss, global_step=global_step)

# Set up adversary
# attack = LinfPGDAttack(model, 
#                        config['epsilon'],
#                        config['k'],
#                        config['a'],
#                        config['random_start'],
#                        config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
# tf.summary.scalar('accuracy train', model.accuracy)
# tf.summary.scalar('accuracy adv', model.accuracy)
# tf.summary.scalar('xent train', model.xent / batch_size)
tf.summary.scalar('bce train', model.bce_loss / batch_size)
# tf.summary.scalar('xent adv', model.xent / batch_size)
# tf.summary.image('images train', model.x_input)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  idxs = np.arange(60000)
  batch_num = int(len(idxs)/batch_size)
  for ii in range(max_num_training_steps):
    _beg = ii % batch_num * batch_size
    _end = min(60000, _beg+batch_size)

    if ii%batch_num == 0:
      idxs = np.random.permutation(idxs)
    
    idx = idxs[_beg:_end]
    x_batch = x_train[idx]
    y_batch = y_train[idx]
    # x_batch, y_batch = mnist.train.next_batch(batch_size)

    # Compute Adversarial Perturbations
    start = timer()
    # x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    # adv_dict = {model.x_input: x_batch_adv,
                # model.y_input: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      # nat_acc, nat_loss = sess.run([model.accuracy, model.xent], feed_dict=nat_dict)
      nat_scores, nat_loss = sess.run([model.bce_score, model.bce_loss], feed_dict=nat_dict)
      # adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      nat_scores = np.array(nat_scores)
      nat_labels = np.zeros(nat_scores.shape).astype(np.float32)
      nat_labels[nat_scores>=0.5] = 1.
      nat_acc = np.sum(np.sum(np.absolute(nat_labels-y_batch), axis=-1) == 0) / batch_num
      print(nat_loss.shape)
      print('Step {}: {} -{}  ({})'.format(ii, _beg, _end, datetime.now()))
      print('    training nat loss {:.6}'.format(np.sum(nat_loss))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      # print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=nat_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=nat_dict)
    end = timer()
    training_time += end - start

  saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)


