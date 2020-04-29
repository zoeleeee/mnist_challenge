import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import sys

conf = sys.argv[-1]
target = int(sys.argv[-2])
x_val = np.load('data/mnist_data.npy')[60000:60010].transpose((0,2,3,1)).astype(np.float32) / 255.
labels = (np.load('data/mnist_labels.npy')[60000:60010]+target)%10
if conf.endswith('.py'):
    from cleverhans.attacks import CarliniWagnerL2
    with open('models/mnist.json') as file:
        json_model = file.read()

    model = keras.models.model_from_json(json_model)
    model.load_weights('models/mnist.h5')
    _labels = keras.utils.to_categorical(labels, num_classes=10)
    advs_label = keras.utils.to_categorical(np.load('non_repeat_advs_targeted_labels.npy'), num_classes=10)
    bapp_params = {
        'y':_labels, 
        'max_iterations':1000,
        'abort_early':True,
        'learning_rate':1e-2,
        'binary_search_steps':9,
        'initial_const':1,
        'clip_min':0, 
        'clip_max':1,
        'batch_size':10,
    }
else:
    from l2_attack import CarliniWagnerL2
    nb_models = int(sys.argv[-3])
    models = []
    for i in range(nb_models):
        with open(conf) as config_file:
            config = json.load(config_file)
        
        idxs = np.arange(len(labels))
        while np.sum(labels[idxs] == labels) != 0:
            if np.min(labels[labels[idxs]==labels]) == np.max(labels[labels[idxs]==labels]):
                idxs = np.random.permutation(np.arange(len(labels)))
            idxs[labels[idxs]==labels] = np.random.permutation(idxs[labels[idxs]==labels])
        image_target = x_val[idxs]
        def custom_loss():
            def loss(y_true, y_pred):
                if config['loss_func'] == 'bce':
                    _loss = keras.losses.BinaryCrossentropy()
                    return _loss(y_true, tf.nn.sigmoid(y_pred))
                elif config['loss_func'] == 'xent':
                    _loss = keras.losses.SparseCategoricalCrossentropy()
                    return _loss(y_true, tf.nn.softmax(y_pred))
            return loss
        model = keras.models.load_model(config['model_dir']+'.h5', custom_objects={ 'custom_loss': custom_loss(), 'loss':custom_loss() }, compile=False)
        conf = conf[:conf.find(conf.split('_')[-1])]+str(config['num_labels']*(i+1))+'.json'
        models.append(model)
    orders = np.load(config['permutation']).astype(np.float32)
    orders /= int(config['permutation'].split('/')[-1].split('_')[1].split('.')[0])-1
    label_rep = np.load('2_label_permutation.npy')[0:config['num_labels']*len(models)].T
    labels = np.array([label_rep[i] for i in labels])
    bapp_params = {
        'y_target':labels, 
        'max_iterations':10000,
        'abort_early':True,
        'learning_rate':10,
        'binary_search_steps':1,
        'initial_const':1,
        'clip_min':0, 
        'clip_max':1,
        'batch_size':10,
        'rnd': orders,
#        'targeted':True,
    }

from cleverhans.utils_keras import KerasModelWrapper
keras.backend.set_learning_phase(0)
sess = keras.backend.get_session()

models = [KerasModelWrapper(model) for model in models]
attack = CarliniWagnerL2(models, sess=sess)
x_adv = attack.generate_np(x_val, **bapp_params)#, y_target=advs_label)
# orig_labs = np.argmax(model.predict(x_val), axis=1)
# new_labs = np.argmax(model.predict(x_adv), axis=1)
l2dist = np.linalg.norm(x_val.reshape(x_val.shape[0], -1)-x_adv.reshape(x_adv.shape[0], -1), axis=-1)
print(np.mean(l2dist), np.max(l2dist), np.min(l2dist))
# print('normal mnist model acc:', np.mean(orig_labs==labels))
# print('advs mnist model acc:', np.mean(new_labs==labels))
# print('advs acc:', new_labs[orig_labs==labels] != labels[orig_labs==labels])
np.save('advs/'+conf[:-5].split('/')[-1]+'_'+str(target)+'_cw2_show.npy', x_adv)

#print('avds acc', np.mean(new_labels == advs_label))
# x_adv = self.attack.generate_np(x_val, max_iterations=100,
#                                     binary_search_steps=3,
#                                     initial_const=1,
#                                     clip_min=-5, clip_max=5,
#                                     batch_size=100, y_target=feed_labs)
