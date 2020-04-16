import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np

with open('models/mnist.json') as file:
    json_model = file.read()

model = keras.models.model_from_json(json_model)
model.load_weights('models/mnist.h5')

data = np.load('data/mnist_data.npy')[60000:].astype(np.float32) / 255.
labels = np.load('data/mnist_labels.npy')[60000:]

preds = model.predict(data)
print('acc:' np.mean(np.argmax(preds, axis=1) == labels))

from cleverhans.attacks import MomentumIterativeMethod
with tf.Session() as sess:
    attack = MomentumIterativeMethod(model, sess=sess)
    for decay_factor in [0.0, 0.5, 1.0]:
        x_adv = self.attack.generate_np(data, eps=0.3, ord=np.inf,
                                      decay_factor=decay_factor,
                                      clip_min=0, clip_max=1.0)

        delta = np.max(np.abs(x_adv - data), axis=1)
        self.assertClose(delta, 0.3)
