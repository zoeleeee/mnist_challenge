"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import load_data
import time
from mpmath import *
mp.dps = 1000
from functools import reduce
import operator

class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func, nb_labels, input_shape, batch_size, params):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    # self.sess = sess
    self.models = models
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start

    # if loss_func == 'xent':
    #   loss = model.xent
    # elif loss_func == 'bce':
    #   loss = model.bce_loss
    # elif loss_func == 'cw':
    #   label_mask = tf.one_hot(model.y_input,
    #                           nb_labels,
    #                           on_value=1.0,
    #                           off_value=0.0,
    #                           dtype=tf.float32)
    #   correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
    #   wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
    #                               - 1e4*label_mask, axis=1)
    #   loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    # else:
    #   print('Unknown loss function. Defaulting to cross-entropy')
    #   loss = model.xent
#    self.input = tf.Variable(np.zeros([batch_size, 28, 28, input_shape[-1]]), dtype=tf.float32, name='input')
#    self.labels = tf.Variable(np.zeros([len(models), batch_size, nb_labels]), dtype=tf.float32, name='labels')
    self.assign_input = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, input_shape[-1]])
    self.assign_labels = tf.placeholder(tf.float32, shape=[len(models), batch_size, nb_labels])

    if loss_func == 'hinge':
        hinge_loss = tf.keras.losses.Hinge()
        loss = tf.reduce_sum([hinge_loss(self.assign_labels[i], model(self.assign_input)) for i, model in enumerate(self.models)])
    elif loss_func == 'bce':
        bce_loss = keras.losses.BinaryCrossentropy()
        loss = tf.reduce_sum([bce_loss(self.assign_labels[i], tf.nn.sigmoid(model(self.assign_input))) for i,model in enumerate(self.models)])
    # self.grad = tf.reduce_sum([tf.gradients(loss, )[0] for m in self.models], 0)
    self.grad = tf.gradients(loss, self.assign_input)
    self.param = np.load('lagrange/lag_'+params.split('/')[1])
    self.inv_param = np.load('lagrange/lag_iter_'+params.split('/')[1])
 #   self.input.assign(self.assign_input)
 #   self.labels.assign(self.assign_labels)
  def grad_perm(self, v, i):
    n, tmp, res = 256, mpf(1), mpf(0)
    for j in range(1,n):
      sign = 1 if (n-1)%2 == j%2 else -1
      res += sign * j * tmp * self.param[i][n-j-1]
      tmp *= v
    return np.sign(res)
    # self.setup = []
    # self.setup.append(self.input.assign(self.assign_input))
    # self.setup.append(self.labels.assign(self.assign_labels))
    # self.init = tf.variables_initializer(var_list=)

  def grad_perm(self, v, i):
    n, tmp, res = 256, mpf(1), mpf(0)
    for j in range(1,n):
      # sign = 1 if (n-1)%2 == j%2 else -1
      res += sign * j * tmp * self.param[i][n-j-1]
      tmp *= v
    return np.sign(res)

  def perturb(self, x_nat, y, org_img, sess, order, targeted=False):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    
    # tmp = copy.deepcopy(org_img)
    # print(np.max(tmp), np.min(tmp), np.median(tmp))
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 1) # ensure valid pixel range
    else:
      x = np.copy(x_nat)
    st = time.time()
    for i in range(self.k):
      print(time.time()-st)
      # _x = [[[order[round(v[0])] for v in c] for c in b] for b in (x*255)]
      _x = np.array([[[[mpf(str(v)) for v in c] for c in b] for b in a] for a in x])
      _x = np.array([[np.polyval(self.param[j], np.squeeze(_x))] for j in range(order.shape[-1])]).transpose((1,2,3,0)).astype(np.float64)
      print(np.max(_x), np.min(_x))
      grad = sess.run(self.grad, feed_dict={self.assign_input:_x, self.assign_labels:y})
      grad = np.array(grad)[0]

      sub_grad = np.array([np.polyval(self.inv_param[j], x) for j in range(_x.shape[-1])]).transpose((1,2,3,0)).astype(np.float64)
      grad = np.sum(sub_grad*grad, axis=-1).reshape(x.shape)

#      print(grad.shape, np.sum(np.sign(grad)==0), np.sum(np.sign(grad)>0))

      # x += self.a * np.sign(grad)
      if targeted:
        x -= self.a*np.sign(grad)

      else:
        x += self.a*np.sign(grad)
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      x = np.clip(x, 0, 1) # ensure valid pixel range      
    x = np.array([[[[round(v[0])] for v in c] for c in b] for b in x*255])
    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  conf = sys.argv[-1]
  nb_models = int(sys.argv[-3])
  targeted = (sys.argv[-2] == 'target')
  loss_func = sys.argv[-4]

  models, y_test = [], []
  lab_permutation = np.load('2_label_permutation.npy')
  if targeted:
    y_lab = np.load('non_repeat_advs_targeted_labels.npy')
  else:
    y_lab = np.load('data/mnist_labels.npy')[60000:]

  loss = ''
  for i in range(nb_models):
    print(conf)
    with open(conf) as config_file:
      config = json.load(config_file)

    def custom_loss():
      def loss(y_true, y_pred):
        if config['loss_func'] == 'bce':
            _loss = keras.losses.BinaryCrossentropy()
            return _loss(y_true, tf.nn.sigmoid(y_pred))
        elif config['loss_func'] == 'xent':
            _loss = keras.losses.SparseCategoricalCrossentropy()
            return _loss(y_true, tf.nn.softmax(y_pred))
        return loss
#   keras.losses.custom_loss = custom_loss
    #model_file = tf.train.latest_checkpoint(config['model_dir'])
    model = keras.models.load_model(config['model_dir']+'.h5', custom_objects={ 'custom_loss': custom_loss(), 'loss':custom_loss() }, compile=False)
    models.append(model)
#    if loss == '':
#        loss = config['loss_func']
#    elif loss != config_file['loss_func']:
#        print('loss func inconsistent')
#        exit()
#
    nb_labels = config['num_labels']

    if config['loss_func'] == 'bce':
      lab_perm = lab_permutation[config['start_label']:config['start_label']+nb_labels].T
      y_test.append([lab_perm[j] for j in y_lab])
    conf = conf[:conf.find(conf.split('_')[-1])]+str(nb_labels*(i+1))+'.json'
  y_test = np.array(y_test).astype(np.float32)
  print(y_test.shape)



  permutation_path = config['permutation']
  path = config['store_adv_path'].split('/')[0] + '/sign_'+config['store_adv_path'].split('/')[1]
  org_imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1)).astype(np.float64)/255.
  # org_labs = np.load('data/mnist_labels.npy')[60000:]
  # imgs, labs, input_shape = load_data(permutation_path)
  x_test = org_imgs[60000:]
  # x_test, y_test = imgs[60000:], labs[60000:]
  if targeted:
    path = path.split('/')[0] + '/target_'+path.split('/')[1]

  orders = np.load(permutation_path).astype(np.float32)
  orders /= int(permutation_path.split('/')[-1].split('_')[1].split('.')[0])-1
  # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  # model = Model(input_shape[-1], nb_labels)
  attack = LinfPGDAttack(models,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         loss_func, #config['loss_func'],
                         nb_labels, orders.shape, config['eval_batch_size'], config['permutation'])  

  # idxs = np.arange(x_test.shape[0])
  with tf.Session() as sess:
#<<<<<<< HEAD
#    sess.run(tf.initialize_all_variables())

#=======
    sess.run(tf.global_variables_initializer())
#>>>>>>> df2b83375f333e49e3c1110078c38addf93fad74
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
      print('{}# batch size: {}'.format(ibatch, bend - bstart))

      x_batch = x_test[bstart:bend, :]
      y_batch = y_test[:,bstart:bend,:]
      org_batch = org_imgs[bstart:bend, :]

      # x_batch_adv, x_batch_show = attack.perturb(x_batch, y_batch, org_batch, sess, orders, targeted)
      x_batch_show = attack.perturb(x_batch, y_batch, org_batch, sess, orders, targeted)

      # x_adv.append(x_batch_adv)
      x_show.append(x_batch_show)

  #  print('Storing examples')
    # x_adv = np.concatenate(x_adv, axis=0)
    # np.save(path, x_adv)
      x_adv = np.concatenate(x_show, axis=0)
      np.save(path[:-10]+'_lag_show.npy', x_adv)
    print('Examples stored in {}'.format(path))
