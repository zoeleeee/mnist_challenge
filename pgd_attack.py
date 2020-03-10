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

  def perturb(self, x_nat, y, org_img, order, sess):
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

      x += self.a * np.sign(grad)
      tmp = np.zeros(org_img.shape)
      for t in range(x.shape[0]):
        for j in range(x.shape[1]):
          for p in range(x.shape[2]):
              min_idx = np.max([0, org_img[t,j,p,0]-int(self.epsilon*255)])
              max_idx = np.min([255, org_img[t,j,p,0]+int(self.epsilon*255)+1])

              _x = np.repeat([x[t,j,p,:]], max_idx-min_idx, axis=0)
              dist = np.sum(np.abs(_x-order[min_idx:max_idx]), axis=-1)
              print(dist.shape)
              tmp[t,j,p,0] = int(np.argmin(dist)+min_idx)
              print(tmp[t,j,p,0])
              x[t,j,p,:] = order[int(tmp[t,j,p,0])]
      

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x, tmp


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  conf = sys.argv[-1]
  permutation_path = config['permutation']
  nb_labels = config['num_labels']

  with open(conf) as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  org_imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1))[60000:]
  # org_labs = np.load('data/mnist_labels.npy')[60000:]
  imgs, labs, input_shape = load_data(permutation_path)
  x_test, y_test = imgs[60000:], labs[60000:]
  orders = np.load(permutation_path).reshape(-1,1)
  # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  model = Model(input_shape[-1], nb_labels)
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

      x_batch_adv, x_batch_show = attack.perturb(x_batch, y_batch, org_batch, orders, sess)

      x_adv.append(x_batch_adv)
      x_show.append(x_batch_show)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    x_adv = np.concatenate(x_show, axis=0)
    np.save(path[:-10]+'show.npy', x_adv)
    print('Examples stored in {}'.format(path))
