import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import sys

conf = sys.argv[-1]
target = int(sys.argv[-2])
x_val = np.load('data/mnist_data.npy')[60000:].transpose((0,2,3,1)).astype(np.float32)
labels = (np.load('data/mnist_labels.npy')[60000:]+target)%10

# num_iter = int(sys.argv[-2])
if conf.endswith('.py'):
    from cleverhans.attacks import ElasticNetMethod
    with open('models/mnist.json') as file:
        json_model = file.read()

    model = keras.models.model_from_json(json_model)
    model.load_weights('models/mnist.h5')
    bapp_params = {
        'max_iterations':100,
        'binary_search_steps':3,
        'initial_const':1,
        'clip_min':0, 
        'clip_max':1,
        'batch_size':10,
    }
else:
    from elastic_net_method import ElasticNetMethod
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
        'max_iterations':100,
        'binary_search_steps':3,
        'initial_const':1,
        'clip_min':0, 
        'clip_max':1,
        'batch_size':10,
        'rnd': orders,
        'y_target':labels,
    }

from cleverhans.utils_keras import KerasModelWrapper
keras.backend.set_learning_phase(0)
sess = keras.backend.get_session()

models = [KerasModelWrapper(model) for model in models]
attack = ElasticNetMethod(models, sess=sess)
x_adv = attack.generate_np(x_val,**bapp_params)
# orig_labs = np.argmax(model.predict(x_val), axis=1)
# new_labs = np.argmax(model.predict(x_adv), axis=1)
l1dist = np.linalg.norm(x_val-x_adv, ord=1, axis=-1)
# l1dist = np.sum(np.absolute(x_adv-x_val, axis=-1))
print(np.mean(l1dist), np.max(l1dist), np.min(l1dist))
# print('normal mnist model acc:', np.mean(orig_labs==labels))
# print('advs mnist model acc:', np.mean(new_labs==labels))
# print('advs acc:', new_labs[orig_labs==labels] != labels[orig_labs==labels])
np.save('advs/'+conf[:-5].split('/')[-1]+'_'+str(num_iter)+'_ead_show.npy', x_adv)

# x_adv = self.attack.generate_np(x_val, max_iterations=100,
#                                     binary_search_steps=3,
#                                     initial_const=1,
#                                     clip_min=-5, clip_max=5,
#                                     batch_size=100, y_target=feed_labs)
