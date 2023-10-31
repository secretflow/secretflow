import torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torch import nn, optim


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt2
import os

from resnet50 import ResNet50
from resnet34 import ResNet34

criterion = nn.CrossEntropyLoss()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s' % device)
    #model = ResNet34().to(device)
    #model.state_dict = torch.load('model/resnet34.pth')
    model = torch.load('model/resnet0050_trip_30.pt')


    transform_validation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    validation_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                   transform=transform_validation)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=100, shuffle=True, num_workers=2)
    length_validation = len(validation_data)

    cost = 0
    correct = 0

    with torch.no_grad():
        for x, y in validation_loader:
            x, y = x.to(device), y.to(device)
            #print(type(x))
            model.eval()
            yhat = model(x)
            yhat = yhat.reshape(-1, 10)
            loss = criterion(yhat, y)
            cost += loss.item()
            _, yhat2 = torch.max(yhat.data, 1)
            correct += (yhat2 == y).sum().item()



    my_loss = cost / len(validation_loader)
    my_accuracy = 100 * correct / length_validation

    print(my_loss)
    print(my_accuracy)

if __name__ == '__main__':
    main()
