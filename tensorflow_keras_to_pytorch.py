from tensorflow import keras
import sys

config = sys.argv[-1]
model = keras.models.load_model(config)
weights = model.get_weights()

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024， 10)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

net = Net()
net.conv1.weight.data=torch.from_numpy(np.transpose(weights[0]))
net.conv1.bias.data=torch.from_numpy(weights[1])
net.conv2.weight.data=torch.from_numpy(np.transpose(weights[2]))
net.conv2.bias.data=torch.from_numpy(weights[3])
net.fc1.weight.data=torch.from_numpy(np.transpose(weights[4]))
net.fc1.bias.data=torch.from_numpy(weights[5])
net.fc2.weight.data=torch.from_numpy(np.transpose(weights[6]))
net.fc2.bias.data=torch.from_numpy(weights[7])

torch.save(net.state_dict(), config[:-2]+'pt')

