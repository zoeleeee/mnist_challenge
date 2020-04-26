import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np

with open('models/mnist.json') as file:
    json_model = file.read()

model = keras.models.model_from_json(json_model)
model.load_weights('models/mnist.h5')

x_val = np.load('data/mnist_data.npy')[60000:60010].transpose((0,2,3,1)).astype(np.float32) / 255.
labels = keras.utils.to_categorical(np.load('data/mnist_labels.npy')[60000:60010])
advs_label = keras.utils.to_categorical(np.load('non_repeat_advs_targeted_labels.npy'), num_classes=10)
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils_keras import KerasModelWrapper
keras.backend.set_learning_phase(0)
sess = keras.backend.get_session()

attack = CarliniWagnerL2(KerasModelWrapper(model), sess=sess)
x_adv = attack.generate_np(x_val, max_iterations=100,
                                    binary_search_steps=3,
                                    initial_const=1,
                                    clip_min=0, clip_max=1,
                                    batch_size=10)#, y_target=advs_label)
orig_labs = np.argmax(model.predict(x_val), axis=1)
new_labs = np.argmax(model.predict(x_adv), axis=1)
l2dist = np.linalg.norm(x_val-x_adv, axis=-1)
print(np.mean(l2dist), np.max(l2dist), np.min(l2dist))
print('normal mnist model acc:', np.mean(orig_labs==labels))
print('advs mnist model acc:', np.mean(new_labs==labels))
print('advs acc:', new_labs[orig_labs==labels] != labels[orig_labs==labels])
#print('avds acc', np.mean(new_labels == advs_label))
# x_adv = self.attack.generate_np(x_val, max_iterations=100,
#                                     binary_search_steps=3,
#                                     initial_const=1,
#                                     clip_min=-5, clip_max=5,
#                                     batch_size=100, y_target=feed_labs)
