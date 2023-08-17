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

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
from torchsummary import summary
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from model import *

writer = SummaryWriter()
losss = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mobilenet().to(device)

transform = transforms.Compose([transforms.ToTensor(),])
dataset = KwenDataset(path = 'dataset', transform= transform)
dataloader = DataLoader(dataset=dataset,batch_size=16,shuffle=True,drop_last=False)

criterion = nn.CrossEntropyLoss().to(device)
loss2 = nn.MSELoss().to(device)
#criterion = nn.MSELoss()
learning_rate = 1
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

decive = 'cuda'
count_idx = 0
for epoch in range(1):
    print(f"epoch : {epoch} ")
    for batch in tqdm.tqdm(dataloader):
        img, label = batch
        optimizer.zero_grad()
        x = img.to(device)
        label = label.to(device)
        output = model(x)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        count_idx += 1
        writer.add_scalar('log', loss, count_idx)
        #print("lr: ", optimizer.param_groups[0]['lr'])
        
        if count_idx%100 == 1 :
            #print("loss: {}     output: {}      label: {}".format(loss.item(), output.item(), label.item()))
            optimizer.param_groups[0]['lr'] *= 0.9
            #print('loss: {}                learning rate: {}                '.format(loss.item(), optimizer.param_groups[0]['lr']))
            print('epoch: {}\t loss: {}'.format(epoch, loss))
            print(torch.argmax(output, dim=1))
            print(label,  optimizer.param_groups[0]['lr'])
            
writer.flush()
writer.close()
print('*'*50,'finish', '*'*50)