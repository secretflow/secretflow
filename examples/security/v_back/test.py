import torch
from    torch.utils.data import DataLoader
from    torchvision import datasets

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import cv2
from matplotlib import image as mping
from matplotlib import  pyplot as plt

from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt2
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import random




model = torch.load('model/resnet0050.pt')

model.linear = torch.nn.AdaptiveAvgPool2d(1)

model.eval()


"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
""
bd = torch.load('./backdoor/bdlab0.pt')
#print(bd.shape)

x0 = mping.imread('./pic/temp1.png')
print(x0.shape)
x0 = x0.transpose(2,0,1)
""
""
x1 = x0
print(x1.shape)
plt.imshow(np.transpose(x1, (1, 2, 0)))
plt.show()
""

""
x0 = torch.tensor(x0).to(device)
x0 = Variable(torch.unsqueeze(x0, dim=0).float(), requires_grad=True)
print(x0.shape)
#print(model(x0).reshape(-1,10).shape)

with torch.no_grad():
    x0[:,:,23:30,23:30] = bd[:,:,23:30,23:30]
#print(model(x0).reshape(-1,10))
""


""
x2 = x0.squeeze(0).detach().clone().to('cpu')
print(x2.shape)
plt.imshow(np.transpose(x2, (1, 2, 0)))
plt.show()
""
"""









criterion = nn.TripletMarginLoss(margin=100,p=2,reduce=False)
#crit = nn.TripletMarginLoss(margin=100,p=2,reduce=False)

#cifardata = torch.zeros(10,5000,3,32,32)
cifardata = torch.load('./data/cifar102.pt')
#count1 = torch.zeros(10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cpu')

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

def oushi(x,y):
    len = len(x)


#利用triplet loss训练后门模型
def main():
    batchsz = 20
    print(cifardata.shape)
    #print(count1.shape)

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    model = torch.load('model/resnet0050.pt')
    model.linear = torch.nn.AdaptiveAvgPool2d(1)

    x, label = iter(cifar_train).next()
    print('x:', x[1].shape, 'label:', label[1])
    abc = label[1].numpy()
    print(abc)
    posi = buildpos(x,label)
    nege,labneg = buildneg(x,label)

    x,posi,nege = x.to(device),posi.to(device),nege.to(device)

    anchor = model(x)
    print(anchor.shape)
    print(len(anchor))
    anchor1 = anchor.reshape(-1,2048)
    print(len(anchor1))
    print(anchor1.shape)
    positive = model(posi)
    positive1 = positive.reshape(-1,2048)
    negetive = model(nege)
    negetive1 = negetive.reshape(-1,2048)

    loss = criterion(anchor,positive,negetive)
    loss1 = criterion(anchor1,positive1,negetive1)
    #print(loss.shape)
    print(loss1)

    ap = torch.norm(anchor1-positive1, p=2, dim=1).to(device)
    an = torch.norm(anchor1-negetive1, p=2, dim=1).to(device)
    ap = torch.mean(ap)
    print(ap.shape)

    aaaaa=torch.zeros(3,5)
    aaaaa[:,:] = 123
    print(aaaaa)



    anchor1 = anchor1.to(device1).detach().numpy()
    positive1 = positive1.to(device1).detach().numpy()
    negetive1 = negetive1.to(device1).detach().numpy()
    po = np.linalg.norm(anchor1[0] - positive1[0], ord=2, axis=None, keepdims=True)
    print(po)
    ne = np.linalg.norm(anchor1[0] - negetive1[0], ord=2, axis=None, keepdims=True)
    print(ne)



    """
    poall = 0
    neall = 0
    
    for i in range(20):
        po1 = np.linalg.norm(anchor1[i] - positive1[i], ord=2, axis=None, keepdims=True)
        print(po1)
        ne1= np.linalg.norm(anchor1[i] - negetive1[i], ord=2, axis=None, keepdims=True)
        poall = poall + po1
        neall = neall + ne1
    """



if __name__ == '__main__':
    main()

