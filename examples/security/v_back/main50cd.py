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

import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cpu')
cifardata = torch.load('./data/cifar102.pt')

def buildpos(x,lable):
    x = x.to(device1)
    lable = lable.to(device1)
    xlen = len(x)
    x1 = torch.zeros(xlen,3,32,32)
    for i in range(xlen):
        a = random.randint(0, 4999)
        lab = lable[i].numpy()
        x1[i] = cifardata[lab,a]
    return x1



def buildneg(x,lable):
    x = x.to(device1)
    lable = lable.to(device1)
    xlen = len(x)
    x2 = torch.zeros(xlen,3,32,32)
    lable2 = torch.zeros(xlen)
    for i in range(xlen):
        a = random.randint(0, 4999)
        lab1 = lable[i].numpy()
        lab = random.randint(0, 9)
        while lab == lab1:
            lab = random.randint(0, 9)
        x2[i] = cifardata[lab,a]
        lable2[i] = lab
    return x2,lable2





def main():
    # Check GPU, connect to it if it is available
    device = ''
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available. GPU will be used for training.")
    else:
        device = 'cpu'

    BEST_ACCURACY = 0
    batchsize = 128

    # Preparing Data
    print("==> Prepairing data ...")
    # Transformation on train data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # transformation on validation data
    transform_validation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download Train and Validation data and apply transformation
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validation_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                   transform=transform_validation)

    # Put data into trainloader, specify batch_size
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=100, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Function to show CIFAR images
    def show_data(image):
        plt.imshow(np.transpose(image[0], (1, 2, 0)), interpolation='bicubic')
        plt.show()

    # show_data(train_data[0])

    # Need to import model a model
    model = ResNet50()
    model.linear = torch.nn.AdaptiveAvgPool2d(1)
    # model = ResNet34()
    # model = CNN_batch()
    # Pass model to GPU

    model = model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    criterion = nn.TripletMarginLoss(margin=30, p=2, reduce=True)
    length_train = len(train_data)
    length_validation = len(validation_data)
    print('Lentrain: ', length_train)
    print('Lenval: ', length_validation)
    print('Lentrainloader: ', len(train_loader))
    num_classes = 10

    # Training
    def train(epochs):
        global BEST_ACCURACY

        BEST_ACCURACY = 0
        dict = {'Train Loss': [], 'Train Acc': [], 'Validation Loss': [], 'Validation Acc': []}
        for epoch in range(epochs):
            print("\nEpoch:", epoch + 1, "/", epochs)
            cost = 0
            correct = 0
            total = 0
            woha = 0

            for i, (x, y) in enumerate(train_loader):
                woha += 1
                model.train()
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                yhat = model(x)
                yhat = yhat.reshape(-1, 2048)

                posi = buildpos(x,y).to(device)
                nege,labneg = buildneg(x,y)
                nege = nege.to(device)
                mposi = model(posi).reshape(-1, 2048)
                mnege = model(nege).reshape(-1, 2048)


                loss = criterion(yhat, mposi,mnege)
                print(i, ':', loss)
                loss.backward()
                optimizer.step()
                cost += loss.item()
            print('epoch:',epoch, 'loss:', loss)

            # Save the model if you get best accuracy on validation data

        torch.save(model, './model/resnet0050_trip_30_cd.pt')
        #torch.save(model, './model/resnet50.pt')
        print("TRAINING IS FINISHED !!!")
        return loss

    results = train(200)

    print(results)


if __name__ == '__main__':
    main()