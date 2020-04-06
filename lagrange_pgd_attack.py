"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from utils import load_data


class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func, nb_labels):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'bce':
      loss = model.bce_loss
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              nb_labels,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                  - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]
    self.param = np.load('lagrange_weights.npy')

  def perturb(self, x_nat, y, org_img, order, sess, targeted=False):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 1) # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})


      # x += self.a * np.sign(grad)
      if targeted:
        x -= self.a*np.sign(grad)

      else:
        x += self.a*np.sign(grad)


      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  conf = sys.argv[-1]
  targeted = (sys.argv[-2] == 'target')

  with open(conf) as config_file:
    config = json.load(config_file)

  permutation_path = config['permutation']
  nb_labels = config['num_labels']
  path = config['store_adv_path'].split('/')[0] + '/lag_'+config['store_adv_path'].split('/')[1]

  lab_perm = np.load('2_label_permutation.npy')[:nb_labels].T

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  org_imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1))[60000:]
  # org_labs = np.load('data/mnist_labels.npy')[60000:]
  imgs, labs, input_shape = load_data(permutation_path)
  x_test, y_test = imgs[60000:], labs[60000:]
  if targeted:
    y_test = np.load('advs_targeted_labels.npy')
    path = path.split('/')[0] + '/target_'+path.split('/')[1]
  if config['loss_func'] == 'bce':
    from multi_model import Model
    model = Model(input_shape[-1], nb_labels)
    y_test = np.array([lab_perm[i] for i in y_test])
  elif config['loss_func'] == 'xent':
    from model import Model
    model = Model(input_shape[-1], nb_labels)

  orders = np.load(permutation_path).astype(np.float32)
  orders /= int(permutation_path.split('/')[-1].split('_')[1].split('.')[0])-1
  # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  # model = Model(input_shape[-1], nb_labels)
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'],
                         nb_labels)
  saver = tf.train.Saver()

  # idxs = np.arange(x_test.shape[0])
  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator
    x_show = []
    print('Iterating over {} batches'.format(num_batches))


    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = x_test[bstart:bend, :]
      y_batch = y_test[bstart:bend]
      org_batch = org_imgs[bstart:bend, :]

      x_batch_adv = attack.perturb(x_batch, y_batch, org_batch, orders, sess, targeted)

      x_adv.append(x_batch_adv)
      # x_show.append(x_batch_show)

    print('Storing examples')
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    # x_adv = np.concatenate(x_show, axis=0)
    # np.save(path[:-10]+'show.npy', x_adv)
    print('Examples stored in {}'.format(path))
