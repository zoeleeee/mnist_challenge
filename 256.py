import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv2d_1 = self.__conv(2, name='conv2d_1', in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.conv2d_2 = self.__conv(2, name='conv2d_2', in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.dense_1 = self.__dense(name = 'dense_1', in_features = 1024, out_features = 1024, bias = True)
        self.dense_2 = self.__dense(name = 'dense_2', in_features = 1024, out_features = 20, bias = True)

    def forward(self, x):
        conv2d_1        = self.conv2d_1(x)
        conv2d_1_activation = F.relu(conv2d_1)
        max_pooling2d_1 = F.max_pool2d(conv2d_1_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2d_2        = self.conv2d_2(max_pooling2d_1)
        conv2d_2_activation = F.relu(conv2d_2)
        max_pooling2d_2 = F.max_pool2d(conv2d_2_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        flatten_1       = max_pooling2d_2.view(max_pooling2d_2.size(0), -1)
        dense_1         = self.dense_1(flatten_1)
        dense_1_activation = F.relu(dense_1)
        dense_2         = self.dense_2(dense_1_activation)
        return dense_2


    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

