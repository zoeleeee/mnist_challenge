import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np

with open('models/mnist.json') as file:
	json_model = file.read()

model = keras.models.model_from_json(json_model)
model.load_weights('models/mnist.h5')

data = np.load('data/mnist_data.npy')[60000:].transpose((0,2,3,1)).astype(np.float32) / 255.
labels = np.load('data/mnist_labels.npy')[60000:]

preds = model.predict(data)
print('acc:', np.mean(np.argmax(preds, axis=1) == labels))

