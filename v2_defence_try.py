import keras
import numpy as np
import json
from utils import extend_data

with open('models/mnist.json') as file:
	json_mnist = file.read()

model = keras.models.model_from_json(json_mnist)
model.load_weights('models/mnist.h5')

data = np.load('data/mnist_data.npy').transpose((0,3,1,2))[60000:]
data =  extend_data('permutation/256_256.1_permutation.npy', data)

pred = model.predict(data)
lab = np.argmax(pred, axis=1)
label = np.load('data/mnist_label.npy')[60000:]

print(np.mean(lab==label))

