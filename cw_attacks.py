## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time
import sys
import json
from mpmath import *
mp.dps = 1000
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Lambda, RepeatVector, Permute
from l2_attack import CarliniL2
# from l0_attack import CarliniL0
# from li_attack import CarliniLi


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

class Model:
    def __init__(sess, restore, param, session=None):
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
        premodel = keras.models.load_model(restore+'.h5', custom_objects={ 'custom_loss': custom_loss(), 'loss':custom_loss() }, compile=False)
        self.param = param
        def lagrange(x):
            
            for i in range(self.param.shape[0]):
                tf.math.polyval(self.param[i], x)
            return 
        def lagrange_output_shape(input_shape):
            return input_shape

        model = Sequential()
        model.add(RepeatVector(param.shape[0]))
        model.add(Permute((2,3,1)))
        model.add(Lambda(lagrange, output_shape=lagrange_output_shape))
        model.add(premodel)
        self.model = model

    def predict(self, data):
        self.model(data)

if __name__ == "__main__":
    import json
    conf = sys.argv[-1]
    with open(conf) as config_file:
        config = json.load(config_file)

    with tf.Session() as sess:
        model = Model()
        data, model =  MNIST(), MNISTModel("models/mnist", sess)
        #data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
        #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
        #                   largest_const=15)

        inputs, targets = generate_data(data, samples=1, targeted=True,
                                        start=0, inception=False)
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        # for i in range(len(adv)):
        #     print("Valid:")
        #     show(inputs[i])
        #     print("Adversarial:")
        #     show(adv[i])
            
        #     print("Classification:", model.model.predict(adv[i:i+1]))

        #     print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)