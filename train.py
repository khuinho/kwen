
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

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os

# display images
from torchvision import utils


# utils
import numpy as np
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import time
import copy

from model import *
from dataset import *

def mobilenet(alpha=2, num_classes=1):
    return MobileNet(alpha, num_classes)

decive = 'cuda'
writer = SummaryWriter()
losss = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mobilenet().to(device)

transform = transforms.Compose([transforms.ToTensor(),])
dataset = KwenDataset(path = 'dataset', transform= transform, len_wqi = 8,lstm = True)

dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size
train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])


train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)


def main():
    model = mobilenet().to(device)


    criterion = nn.MSELoss().to(device)
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    count_idx = 0
    count_idx2 = 0

    model = model.train()
    for epoch in tqdm.tqdm(range(1000)):

        cost = 0.0
        model.train()

        for batch in train_dataloader:
            count_idx +=1

            img = batch[0]
            wqi =batch[1]
            label = batch[2]

            label = label.to(torch.float32)
            label = label.to(device)

            x_= torch.stack(wqi).to(device)
            x_ = torch.transpose(x_,0,1)
            x_ = x_.to(torch.float32)

            x = img
            x = x.to(device)

            data = [x, x_]
            output = model(data)
            output = output.view(-1)

            loss = criterion(output, label)

            loss = loss.to(torch.float32)
            loss.backward()

            optimizer.step()
            #scheduler.step(loss)

            #print('Loss: ', loss, 'epoch: ', epoch, "lr: ", optimizer.param_groups[0]['lr'])
            #print('output: ', output[0:5])
            #print('label : ', label[0:5])
            if count_idx%30 == 1 :
                optimizer.param_groups[0]['lr'] *= 0.99

            if count_idx%500 ==1 and count_idx >100:

                writer.add_scalar('train_loss', loss, epoch)
                print('train:\t',loss, epoch)
                print('train:{}////{}'.format(output[:4],label[:4]))
                break
            
        with torch.no_grad():
            model.eval()
            for batch_v in validation_dataloader:
                count_idx2 +=1
                img = batch_v[0]
                wqi =batch_v[1]
                label = batch_v[2]

                label = label.to(torch.float32)
                label = label.to(device)

                x_= torch.stack(wqi).to(device)
                x_ = torch.transpose(x_,0,1)
                x_ = x_.to(torch.float32)

                x = img
                x = x.to(device)

                data = [x, x_]
                output = model(data)
                output = output.view(-1)

                loss = criterion(output, label)

                loss = loss.to(torch.float32)
                if count_idx2%100 ==1 and count_idx2 >100:          
                    writer.add_scalar('validation_loss', loss, epoch)
                    print('evaluate:\t', loss, epoch)
                    break
                
    torch.save(model, os.path.join('model_save', 'model.pth'))
    
if __name__ == '__main__':
    main()