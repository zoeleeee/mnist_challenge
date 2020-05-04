import keras
import numpy as np
import json
#from utils import extend_data

def extend_data(order, imgs):
	if np.max(imgs) <= 1:
		imgs *= 255
	imgs = imgs.astype(np.int)
	samples = np.array([[[order[d[0]] for d in c] for c in b] for b in imgs]).astype(np.float32) /255.
	return samples


with open('models/mnist.json') as file:
	json_mnist = file.read()

model = keras.models.model_from_json(json_mnist)
model.load_weights('models/mnist.h5')

data = np.load('data/mnist_data.npy').transpose((0,2,3,1))[60000:]
label = np.load('data/mnist_labels.npy')[60000:]


res = np.zeros(label.shape)
preds = []
for i in range(101):
	order = np.random.permutation(np.arange(256))
	data =  extend_data(order.reshape(-1, 1), data)
	pred = model.predict(data)
	preds.append(pred)

score = np.argmax(preds, axis=-1)
lab = [np.argmax(np.bincount(score[:,i] for i in range(score.shape[-1])))]

print(np.mean(np.array(lab)-label))

