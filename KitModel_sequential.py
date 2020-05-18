import torch
import imp
import sys
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 10)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc3(x)
        return x

if __name__ == '__main__':
	net_file = sys.argv[-1]

	imp.load_source('MainModel', 'models/natural.py')
	net = torch.load(net_file)
	weights = []
	for child in net.children():
		for param in list(child.parameters()):
			weights.append(param)
	net = Net()
	net.conv1.weight.data=weights[0]
	net.conv1.bias.data=weights[1]
	net.conv2.weight.data=weights[2]
	net.conv2.bias.data=weights[3]
	net.fc1.weight.data=weights[4]
	net.fc1.bias.data=weights[5]
	net.fc2.weight.data=weights[6]
	net.fc2.bias.data=weights[7]

	torch.save(net.state_dict(), net_file)