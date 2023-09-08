
import glob
import tqdm
import torch
import json
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tensorboard


# import package

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim


# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
import os
from torchsummary import summary


# lstm 분류기 
class RNNNet(nn.Module):
    def __init__(self, input_size,num_classes, hidden_size = 16, init_weights=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = True
        self.rnn = torch.nn.LSTM(self.input_size, self.hidden_size, dropout = 0, batch_first = True, bidirectional = self.bidirectional)
        self.linear3 = nn.Linear(self.input_size*(int(self.bidirectional)+1), num_classes)

    def forward(self, x):
        x, (hidden_state, cell_state) = self.rnn(x)
        x = self.linear3(x)
        return x
    




class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1) 
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x

# Depthwise Separable Convolution
class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

        self.seblock = SEBlock(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.seblock(x) * x
        return x


# BasicConv2d
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# MobileNetV1
class MobileNet(nn.Module):
    def __init__(self, width_multiplier, num_classes, init_weights=True):
        super().__init__()
        self.init_weights=init_weights
        alpha = width_multiplier
        
        self.num_classes = num_classes

        self.conv1 = BasicConv2d(3, int(32*alpha), 3, stride=2, padding=1)
        self.conv2 = Depthwise(int(32*alpha), int(64*alpha), stride=1)
        # down sample
        self.conv3 = nn.Sequential(
            Depthwise(int(64*alpha), int(128*alpha), stride=2),
            Depthwise(int(128*alpha), int(128*alpha), stride=1)
        )
        # down sample
        self.conv4 = nn.Sequential(
            Depthwise(int(128*alpha), int(256*alpha), stride=2),
            Depthwise(int(256*alpha), int(256*alpha), stride=1)
        )
        # down sample
        self.conv5 = nn.Sequential(
            Depthwise(int(256*alpha), int(512*alpha), stride=2),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
        )
        # down sample
        self.conv6 = nn.Sequential(
            Depthwise(int(512*alpha), int(1024*alpha), stride=2)
        )
        # down sample
        self.conv7 = nn.Sequential(
            Depthwise(int(1024*alpha), int(1024*alpha), stride=2)
        )
        # linear
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(int(1024*alpha), 256)
        self.linear2 = nn.Linear( 256, 16)


        # lstm
        self.input_size = 256
        self.hidden_size = 8
        self.bidirectional = True
        self.rnn = torch.nn.LSTM(self.input_size, self.hidden_size, dropout = 0, batch_first = True, bidirectional = self.bidirectional)
        self.lstm = RNNNet(input_size=8, num_classes=4, hidden_size = 8, init_weights=True)
        self.linear3 = nn.Linear(16*(int(self.bidirectional)+1), self.num_classes)
        self.linear4 = nn.Linear(20,self.num_classes)
        
        # weights initialization
        if self.init_weights:
            self._initialize_weights()

    def forward(self, data):
       
        x_ = data[1] 
        x_ = self.lstm(x_)
        
        
        x = data[0]
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.linear2(x)
        x = torch.cat((x, x_), dim=1)
        x = self.linear4(x)
        return x
    
    # weights initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class testNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(32,self.num_classes)
        
    def forward(self, x):
        x = self.linear(x)
        return x
        
def mobilenet(alpha=1, num_classes=1):
    return MobileNet(alpha, num_classes)
            
