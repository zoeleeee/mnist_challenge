import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np

#with open('models/mnist.json') as file:
#    json_model = file.read()

#model = keras.models.model_from_json(json_model)
#model.load_weights('models/mnist.h5')
model = keras.models.load_model('models/natural.h5')
data = np.load('data/mnist_data.npy')[60000:].transpose((0,2,3,1)).astype(np.float32) / 255.
labels = np.load('data/mnist_labels.npy')[60000:]

preds = model.predict(data)
print('acc:', np.mean(np.argmax(preds, axis=1) == labels))

from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper
#with tf.Session() as sess:
keras.backend.set_learning_phase(0)
sess = keras.backend.get_session()

attack = MomentumIterativeMethod(KerasModelWrapper(model), sess=sess)
for decay_factor in [.5]:
    x_adv = attack.generate_np(data, eps=0.3, ord=np.inf,
                                      decay_factor=decay_factor,
                                      clip_min=0, clip_max=1.0)
    print('advs acc:', np.mean(np.argmax(model.predict(x_adv), axis=1)==labels))
    delta = np.max(np.max(np.abs(x_adv - data), axis=1))
    print(delta)
    np.save('advs/normal_mnist_MIM_advs_{}_show.npy'.format(decay_factor), np.clip(x_adv*255., 0, 255))
