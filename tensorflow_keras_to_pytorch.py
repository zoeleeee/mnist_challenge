from tensorflow import keras as k
import sys
import numpy as np
config = sys.argv[-1]
loss_func = 'bce'
input_shape = eval(sys.argv[-3])
nb_channel = int(sys.argv[-2])
def custom_loss():
  def loss(y_true, y_pred):
    if loss_func == 'bce':
      _loss = keras.losses.BinaryCrossentropy()
      return _loss(y_true, tf.nn.sigmoid(y_pred))
    elif loss_func == 'xent':
      _loss = keras.losses.SparseCategoricalCrossentropy()
      return _loss(y_true, tf.nn.softmax(y_pred))
    elif loss_func == 'balance':
      y_true[y_true==0]=-1
      return -1*np.sum(y_true*(y_pred-.5))
  return loss
model = k.models.load_model(config, custom_objects={ 'custom_loss': custom_loss(), 'loss':custom_loss() }, compile=False)
weights = model.get_weights()

import keras

model = keras.Sequential([keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=input_shape, padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(20)
    ])

model.set_weights(weights)
model.save(config)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(32, 32, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 5)
#         self.fc1 = nn.Linear(3136, 1024)
#         self.fc2 = nn.Linear(1024, 20)
#         # self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 1024)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         # x = self.fc3(x)
#         return x

# net = Net()

# print(weights[0].shape)
# print(net.conv1.weight.data.shape)


# net.conv1.weight.data=torch.from_numpy(np.transpose(weights[0], (3,2,0,1)))
# net.conv1.bias.data=torch.from_numpy(weights[1])
# net.conv2.weight.data=torch.from_numpy(np.transpose(weights[2], (3,2,0,1)))
# net.conv2.bias.data=torch.from_numpy(weights[3])
# net.fc1.weight.data=torch.from_numpy(np.transpose(weights[4]))
# net.fc1.bias.data=torch.from_numpy(weights[5])
# net.fc2.weight.data=torch.from_numpy(np.transpose(weights[6]))
# net.fc2.bias.data=torch.from_numpy(weights[7])

# #torch.save(net.state_dict(), config[:-2]+'pt')

# from utils import load_data
# imgs, labs, _ = load_data('permutation/256_256.32_permutation.npy', 10)
# imgs = imgs.transpose((0,3,1,2))[60000:60100]
# imgs = torch.clamp(torch.tensor(imgs), 0, 1)
# net.eval()
# preds = net(imgs)
# np.save('preds/pytorch_test.npy', preds.detach().numpy())

# preds = model.predict(imgs.numpy().transpose((0,2,3,1)))
# np.save('preds/keras_test.npy', preds)
