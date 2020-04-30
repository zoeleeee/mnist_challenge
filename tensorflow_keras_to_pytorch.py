from tensorflow import keras
import sys
import numpy as np
config = sys.argv[-1]
model = keras.models.load_model(config)
weights = model.get_weights()

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 20)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc3(x)
        return x

net = Net()

print(weights[0].shape)
print(net.conv1.weight.data.shape)


net.conv1.weight.data=torch.from_numpy(np.transpose(weights[0], (3,2,0,1)))
net.conv1.bias.data=torch.from_numpy(weights[1])
net.conv2.weight.data=torch.from_numpy(np.transpose(weights[2], (3,2,0,1)))
net.conv2.bias.data=torch.from_numpy(weights[3])
net.fc1.weight.data=torch.from_numpy(np.transpose(weights[4]))
net.fc1.bias.data=torch.from_numpy(weights[5])
net.fc2.weight.data=torch.from_numpy(np.transpose(weights[6]))
net.fc2.bias.data=torch.from_numpy(weights[7])

torch.save(net.state_dict(), config[:-2]+'pt')

from utils import load_data
imgs, labs, _ = load_data('permutation/256_256.16_permutation.npy', 10)
imgs = imgs.transpose((0,3,1,2))[60000:60100]
imgs = torch.clamp(torch.tensor(imgs), 0, 1)
net.eval()
preds = net(imgs)
np.save('preds/pytorch_test.npy', preds.detach().numpy())
