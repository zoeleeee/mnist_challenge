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


class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func, nb_labels, input_shape):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
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
    
    self.input = tf.placeholder(tf.float32, shape=[None, 28, 28, input_shape[-1]])
    self.labels = tf.placeholder(tf.float32, shape=[len(models), None, nb_labels])
    bce_loss = keras.losses.BinaryCrossentropy()
    loss = tf.reduce_sum([bce_loss(self.labels[i], tf.nn.sigmoid(model.predict(self.input))) for i,model in enumerate(self.models)])
    # self.grad = tf.reduce_sum([tf.gradients(loss, )[0] for m in self.models], 0)
    self.grad = tf.gradients(loss, self.input)


  def perturb(self, x_nat, y, org_img, order, sess, targeted=False):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 1) # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      # tt = dict([(m.x_input,x) for m in self.models]+[(self.models[i].y_input:y[i] for i in range(len(self.models)))])
      grad = sess.run(self.grad, feed_dict={self.input:x, self.labels:y})
      print(grad.shape, np.sum(np.sign(grad!=0)), np.sum(np.sign(grad>0)))

      # x += self.a * np.sign(grad)
      if targeted:
        x_upd = x-self.a*np.sign(grad)

      else:
        x_upd = x+self.a*np.sign(grad)

      tmp = np.zeros(org_img.shape)
      for t in range(x.shape[0]):
        for j in range(x.shape[1]):
          for p in range(x.shape[2]):
              min_idx = np.max([0, org_img[t,j,p,0]-int(self.epsilon*255)])
              max_idx = np.min([255, org_img[t,j,p,0]+int(self.epsilon*255)+1])

              if targeted:
                sign_neighbors = np.sign(-order[min_idx:max_idx]+np.repeat([x[t,j,p,:]], max_idx-min_idx, axis=0))
              else:
                sign_neighbors = np.sign(order[min_idx:max_idx]-np.repeat([x[t,j,p,:]], max_idx-min_idx, axis=0))
              matches = np.sum(np.repeat([np.sign(grad[t,j,p,:])], max_idx-min_idx, axis=0) == sign_neighbors, axis=-1)

              idxs = np.arange(len(matches))[matches==np.max(matches)]
              _x = np.repeat([x_upd[t,j,p,:]], len(idxs), axis=0)
              dist = np.sum(np.abs(_x-order[idxs+min_idx]), axis=-1)
              # print(dist.shape)

              tmp[t,j,p,0] = int(idxs[np.argmin(dist)]+min_idx)
              # print(tmp[t,j,p,0])
              x[t,j,p,:] = order[int(tmp[t,j,p,0])]

      # x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      # x = np.clip(x, 0, 1) # ensure valid pixel range

    return x, tmp


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  conf = sys.argv[-1]
  nb_models = int(sys.argv[-3])
  targeted = (sys.argv[-2] == 'target')

  models, y_test = [], []
  lab_permutation = np.load('2_label_permutation.npy')
  if targeted:
    y_lab = np.load('advs_targeted_labels.npy')
  else:
    y_lab = np.load('data/mnist_labels.npy')[60000:]

  loss = ''
  for i in range(nb_models):
    with open(conf) as config_file:
      config = json.load(config_file)

    def custom_loss(y_true, y_pred):
    if config['loss_func'] == 'bce':
        loss = keras.losses.BinaryCrossentropy()
        return loss(y_true, y_pred)
    elif config['loss_func'] == 'xent':
        loss = keras.losses.SparseCategoricalCrossentropy()
        return loss(y_true, keras.activations.softmax(y_pred))

    model_file = tf.train.latest_checkpoint(config['model_dir'])
    model = keras.models.load_model(config['model_dir']+'.h5', , custom_objects={ 'loss': custom_loss })
    models.append(model)
    if loss == '':
        loss = config['loss_func']
    elif loss != config_file['loss_func']:
        print('loss func inconsistent')
        exit()

    nb_labels = config['num_labels']

    if loss == 'bce':
      lab_perm = lab_permutation[config['start_label']:config['start_label']+nb_labels]
      y_test.append([lab_perm[i] for i in y_lab])
    conf = conf[:-6]+str(nb_labels*(i+1))+'.json'
  y_test = np.array(y_test).astype(np.float32)
  print(y_test.shape)



  permutation_path = config['permutation']
  path = config['store_adv_path'].split('/')[0] + '/sign_'+config['store_adv_path'].split('/')[1]
  org_imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1))[60000:]
  # org_labs = np.load('data/mnist_labels.npy')[60000:]
  imgs, labs, input_shape = load_data(permutation_path)
  x_test = imgs[60000:]
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
                         config['loss_func'],
                         nb_labels)

  # idxs = np.arange(x_test.shape[0])
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Iterate over the samples batch-by-batch
    num_eval_examples = 20#config['num_eval_examples']
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
      y_batch = y_test[:,bstart:bend,:]
      org_batch = org_imgs[bstart:bend, :]

      x_batch_adv, x_batch_show = attack.perturb(x_batch, y_batch, org_batch, orders, sess, targeted)

      x_adv.append(x_batch_adv)
      x_show.append(x_batch_show)

    print('Storing examples')
    # x_adv = np.concatenate(x_adv, axis=0)
    # np.save(path, x_adv)
    x_adv = np.concatenate(x_show, axis=0)
    np.save(path[:-10]+'show.npy', x_adv)
    print('Examples stored in {}'.format(path))
