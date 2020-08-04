#CUDA_VISIBLE_DEVICES=0 python keras_rnd_multi_eval.py 0.9 window 16 1 100 0 4 configs/

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

from utils import *
import numpy as np

conf = sys.argv[-1]
nb_models = int(sys.argv[-2])
t = int(sys.argv[-3])
nb_imgs = int(sys.argv[-4])
st_imgs = int(sys.argv[-5])
input_bytes = eval(sys.argv[-6])
_type = sys.argv[-7]
_t = eval(sys.argv[-8])
#dataset = sys.argv[-2]
# Global constants
with open(conf) as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']
nb_labels = config['num_labels']
st_lab = config['start_label']
rep = np.load('2_label_permutation.npy')[st_lab:st_lab+nb_labels*nb_models].T
rep[rep==0] = -1
nb_channel = int(config['permutation'].split('_')[1].split('.')[1])
nb_label = config['num_labels']
#if dataset == 'origin.npy':
# imgs, labels, input_shape = load_data(config['permutation'], config['num_labels'])
labels = np.load('data/mnist_labels.npy')
imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1))
permut = np.load(config['permutation'])
# labels = np.array([rep[i] for i in labels]).astype(np.float32)
x_train, y_train = imgs[:60000], labels[:60000]
x_test, y_test = imgs[-nb_imgs-st_imgs:-st_imgs], labels[-nb_imgs-st_imgs:-st_imgs]

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

models = []
if _type == 'window':
    model_var = '_window' + str(input_bytes)
  elif _type == 'slide4':
    model_var = '_slide'+str(input_bytes)
for i in range(nb_models):
  with open(conf) as config_file:
    config = json.load(config_file)
  model_dir = config['model_dir']
  models.append(keras.models.load_model(model_dir+model_var+'.h5', custom_objects={ 'custom_loss': custom_loss(), 'loss':custom_loss() }, compile=False))
  conf = conf[:conf.find(conf.split('_')[-1])]+str(nb_labels*(i+1))+'.json'

tot_advs_acc = np.zeros(len(y_test))
tot_amt = 0
change_advs_acc = []
rnd_imgs = np.zeros(imgs[-nb_imgs-st_imgs:-st_imgs].shape)
print(rnd_imgs.shape, x_test.shape)
while True:
  if np.mean(tot_advs_acc) == 1.: 
    print(tot_amt, 'totally attacked succeed!')
    np.save('preds/rnd_'+model_dir.split('/')[-1]+'.npy', change_advs_acc)
    break
  elif tot_amt == 1e5:
    np.save('preds/rnd_'+model_dir.split('/')[-1]+'.npy', change_advs_acc)
    print(tot_amt, 'total adversarial acc:', tot_advs_acc)
    break
  else:
    tot_amt += 1
    noise = x_test
    # noise = np.clip(np.random.randint(-1*int(config['epsilon']*255), int(config['epsilon']*255), x_test.shape)+x_test, 0, 255).astype(np.int)
    if _type == 'window':
      samples = [window_perm_sliding_img_AES(nb_channel, noise, i*nb_label, input_bytes) for i in range(nb_models)]
    if _type == 'slide4':
      samples = [four_pixel_perm_sliding_img_AES(nb_channel, noise, i*nb_label, input_bytes) for i in range(nb_models)]
    # samples = np.array([[[permut[d[0]] for d in c] for c in b] for b in noise])
    x_input = [samples[i].astype(np.float32) / 255. for i in range(len(models))]
    scores = []
    for i in range(nb_models):
      scores.append(models[i].predict(x_input[i], batch_size=eval_batch_size))
    scores = np.hstack(scores)
    nat_labels = np.zeros(scores.shape)
    nat_labels[scores>=_t] = 1.
    if _t == .5:
      nat_labels[scores<1-_t] = -1
    else:
      nat_labels[scores <= 1-_t] = -1

    preds, preds_dist, preds_score = [], [], []
    print(scores.shape)
    for i in range(len(nat_labels)):
      tmp = np.repeat([nat_labels[i]], rep.shape[0], axis=0)
      dists = np.sum(np.absolute(tmp-rep), axis=-1)
      min_dist = np.min(dists)
      pred_labels = np.arange(len(dists))[dists==min_dist]
      pred_scores = [np.sum([scores[i][k] if rep[j][k]==1 else 1-scores[i][k] for k in np.arange(len(scores[i]))]) for j in pred_labels]
      pred_label = pred_labels[np.argmax(pred_scores)]
      preds.append(pred_label)
      preds_dist.append(dists[pred_label])
      preds_score.append(np.max(pred_scores))
    error_idxs = np.arange(len(preds))[preds != y_test]
    preds = np.array(preds)
    preds_dist = np.array(preds_dist)
    tot_advs_acc[error_idxs[preds_dist[preds!=y_test]<= t]] = 1.
    print(rnd_imgs.shape, noise.shape)
    rnd_imgs[error_idxs[preds_dist[preds!=y_test]<= t]] = noise[error_idxs[preds_dist[preds!=y_test]<= t]]
    change_advs_acc.append(np.mean(tot_advs_acc))
    if tot_amt % 1000 == 0:
      np.save('advs/rnd_'+model_dir.split('/')[-1]+'.npy', rnd_imgs)
    print('{} natural: {:.2f}%; total adversarial acc:{}'.format(tot_amt, np.sum(preds_dist[preds!=y_test] <= t), np.mean(tot_advs_acc)))
