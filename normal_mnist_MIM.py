import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import sys


decay_factor = eval(sys.argv[-1])
eps_iter = eval(sys.argv[-2])
nb_iter = eval(sys.argv[-3])
eps = eval(sys.argv[-4])
sign = sys.argv[-5]
order = eval(sys.argv[-6])
attack_method = sys.argv[-7]

if order == -1:
	order = np.inf

#with open('models/mnist.json') as file:
#    json_model = file.read()

#model = keras.models.model_from_json(json_model)
#model.load_weights('models/mnist.h5')
model = keras.models.load_model('models/natural.h5')
idxs = np.arange(10000)#np.load('data/final_random_1000_correct_idxs.npy')
data = np.load('data/mnist_data.npy')[60000:]
data = data[idxs].transpose((0,2,3,1)).astype(np.float32) / 255.
labels = np.load('data/mnist_labels.npy')[60000:]
labels = labels[idxs]

# preds = model.predict(data)
# np.save('preds/nat_predict.npy', np.argmax(preds, axis=1))
# print('acc:', np.mean(np.argmax(preds, axis=1) == labels))

from cleverhans.attacks import MomentumIterativeMethod, CarliniWagnerL2, ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper
#with tf.Session() as sess:
keras.backend.set_learning_phase(0)
sess = keras.backend.get_session()

if attack_method == 'MIM':
  attack = MomentumIterativeMethod(KerasModelWrapper(model), sess=sess)
  params = {'eps': eps,
                'nb_iter': nb_iter,
                'eps_iter': eps_iter,
                'ord': order,
                'decay_factor': decay_factor,
                'clip_max': 1.,
                'clip_min': 0}
elif attack_method == 'PGD':
  attack = ProjectedGradientDescent(KerasModelWrapper(model), sess=sess)
  params = {'eps': eps, 
              'eps_iter': eps_iter,
              'nb_iter': nb_iter,
              'clip_max': 1.,
              'clip_min': 0}
elif attack_method == 'CW':
  attack = CarliniWagnerL2(KerasModelWrapper(model), sess=sess)
  params = {'confidence': eps}
  data = data[eps_iter:eps_iter+decay_factor]
  labels = labels[eps_iter:eps_iter+decay_factor]
# generate adversarial examples


# attack = MomentumIterativeMethod(KerasModelWrapper(model), sess=sess)
# for decay_factor in [.5]:
x_advs = []
succ_idxs = []
sep = 10
for i in range(0, len(data), sep):
  x_adv = attack.generate_np(data[i:i+sep], **params)
  x_adv = (np.array(x_adv)*255.).astype(np.uint8)

  attack_sign = np.argmax(model.predict(x_adv.astype(np.float32)/255.), axis=1)!=labels
  x_advs.append(x_adv[attack_sign])
  succ_idxs.append(np.arange(i, i+sep)[attack_sign])

print('model acc under aes:', 1-len(idxs)/len(data))
np.save('advs/mnist_'+attack_method+'_'+sign+'_advs_show.npy', np.vstack(x_advs))
np.save('advs/mnist_'+attack_method+'_'+sign+'_advs_label.npy', labels[succ_idxs])
np.save('advs/mnist_'+attack_method+'_'+sign+'_advs_label.npy', np.hstack(succ_idxs))
