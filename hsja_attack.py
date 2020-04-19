import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import sys

x_val = np.load('data/mnist_data.npy')[60000:].transpose((0,2,3,1)).astype(np.float32) / 255.
labels = np.load('data/mnist_labels.npy')[60000:]
conf = sys.argv[-1]
if conf.endswith('.py'):
    from cleverhans.attacks import HopSkipJumpAttack
    with open('models/mnist.json') as file:
        json_model = file.read()
    model = keras.models.model_from_json(json_model)
    model.load_weights('models/mnist.h5')
    bapp_params = {
        'constraint': 'linf',
        'stepsize_search': 'geometric_progression',
        'num_iterations': 10,
        'verbose': True,
    }
else:    
    from hop_skip_jump_attack import HopSkipJumpAttack
    nb_models = int(sys.argv[-2])
    models = []
    for i in range(nb_models):
        with open(conf) as config_file:
            config = json.load(config_file)
        label_rep = rep = np.load('2_label_permutation.npy')[config['start_label']:config['start_label']+config['num_labels']*len(models)].T
        idxs = np.arange(len(labels))
        while np.sum(idxs == labels) != 0:
            idxs[idxs==labels] = np.random.permutation(idxs[idxs==labels])
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
        conf = conf[:conf.find(conf.split('_')[-1])]+str(nb_labels*(i+1))+'.json'
        models.append(model)
    bapp_params = {
        'constraint': 'linf',
        'stepsize_search': 'geometric_progression',
        'num_iterations': 10,
        'verbose': True,
        'original_label': labels,
        'label_rep': label_rep,
        'image_target': image_target,
        'clip_min': 0,
        'clip_max':255,
    }


from cleverhans.utils_keras import KerasModelWrapper
keras.backend.set_learning_phase(0)
sess = keras.backend.get_session()

models = [KerasModelWrapper(model) for model in models]
attack = HopSkipJumpAttack(models, sess=sess)

x_adv = attack.generate_np(x_val, **bapp_params)
orig_labs = np.argmax(model.predict(x_val), axis=1)
new_labs = np.argmax(model.predict(x_adv), axis=1)
print(np.max(np.absolute(x_adv-x_val)))
print('normal mnist model acc:', np.mean(orig_labs==labels))
print('advs mnist model acc:', np.mean(new_labs==labels))
print('advs acc:', new_labs[orig_labs==labels] != labels[orig_labs==labels])

# from hop_skip_jump_attack import HopSkipJumpAttack
# import json
# import tensorflow as tf 
# from tensorflow import keras
# import numpy as np
# from cleverhans.utils_keras import KerasModelWrapper
# from utils import 

# attack = HopSkipJumpAttack(KerasModelWrapper(model), sess)
# x_adv = np.load('data/mnist_data.npy').transpose((0,2,3,1))

# bapp_params = {
#         'constraint': 'l2',
#         'stepsize_search': 'geometric_progression',
#         'num_iterations': 10,
#         'verbose': True,
#     }
# x_adv = self.attack.generate_np(x_val, **bapp_params)

# bapp_params = {
#         'constraint': 'linf',
#         'stepsize_search': 'grid_search',
#         'num_iterations': 10,
#         'verbose': True,
#     }
# bapp_params = {
#         'constraint': 'linf',
#         'stepsize_search': 'geometric_progression',
#         'num_iterations': 10,
#         'verbose': True,
#         'y_target': y_target,
#         'image_target': image_target,
#     }
# bapp_params = {
#         'constraint': 'l2',
#         'stepsize_search': 'grid_search',
#         'num_iterations': 10,
#         'verbose': True,
#         'y_target': y_target_ph,
#         'image_target': image_target_ph,
#     }
