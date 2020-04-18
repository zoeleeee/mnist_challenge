import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np

with open('models/mnist.json') as file:
    json_model = file.read()

model = keras.models.model_from_json(json_model)
model.load_weights('models/mnist.h5')

x_val = np.load('data/mnist_data.npy')[60000:].transpose((0,2,3,1)).astype(np.float32) / 255.
labels = np.load('data/mnist_labels.npy')[60000:]

from cleverhans.attacks import HopSkipJumpAttack
from cleverhans.utils_keras import KerasModelWrapper
keras.backend.set_learning_phase(0)
sess = keras.backend.get_session()

attack = ElasticNetMethod(KerasModelWrapper(model), sess=sess)
x_adv = attack.generate_np(x_val, max_iterations=100,
                                    binary_search_steps=3,
                                    initial_const=1,
                                    clip_min=-5, clip_max=5,
                                    batch_size=10)
orig_labs = np.argmax(model.predict(x_val), axis=1)
new_labs = np.argmax(model.predict(x_adv), axis=1)
l1dist = np.sum(np.absolute(x_adv-x_val, axis=-1))
print(np.mean(l1dist), np.max(l1dist), np.min(l1dist))
print('normal mnist model acc:', np.mean(orig_labs==labels))
print('advs mnist model acc:', np.mean(new_labs==labels))
print('advs acc:', new_labs[orig_labs==labels] != labels[orig_labs==labels])