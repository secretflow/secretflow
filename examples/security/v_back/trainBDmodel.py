# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import matplotlib.pylab as plt2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from torchvision import datasets

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random

from imageio import imsave
from resnet34 import ResNet34
from resnet50 import ResNet50
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cpu")


def tripletloss(anchor1, positive1, negetive1, margin, t):
    margin = torch.tensor(margin).to(device)
    t = torch.tensor(t).to(device)
    ap = torch.norm(anchor1 - positive1, p=2, dim=1).to(device)
    an = torch.norm(anchor1 - negetive1, p=2, dim=1).to(device)
    ap1 = torch.mean(ap).to(device)
    an1 = torch.mean(an).to(device)

    # t = t.to(device)
    loss = t * ap1 - an1 + margin
    return loss


criterion = nn.TripletMarginLoss(margin=20, p=2, reduce=True)


# criterion = nn.CrossEntropyLoss()
# mse_loss = torch.nn.MSELoss(reduce=False, size_average=False)
def imshow(img):
    # img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# cifardata = torch.zeros(10,5000,3,32,32)
cifardata = torch.load("./data/cifar102.pt")
# count1 = torch.zeros(10)


def buildpos(x, lable):
    x = x.to(device1)
    lable = lable.to(device1)
    xlen = len(x)
    x1 = torch.zeros(xlen, 3, 32, 32)
    for i in range(xlen):
        a = random.randint(0, 4999)
        lab = lable[i].numpy()
        x1[i] = cifardata[lab, a]
    return x1


def buildneg(x, lable):
    x = x.to(device1)
    lable = lable.to(device1)
    xlen = len(x)
    x2 = torch.zeros(xlen, 3, 32, 32)
    lable2 = torch.zeros(xlen)
    for i in range(xlen):
        a = random.randint(0, 4999)
        lab1 = lable[i].numpy()
        lab = random.randint(0, 9)
        while lab == lab1:
            lab = random.randint(0, 9)
        x2[i] = cifardata[lab, a]
        lable2[i] = lab
    return x2, lable2


# 利用triplet loss训练后门模型
def main():
    batchsz = 128
    print(cifardata.shape)
    # print(count1.shape)

    cifar_train = datasets.CIFAR10(
        "cifar",
        True,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        download=True,
    )
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10(
        "cifar",
        False,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        download=True,
    )
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print("x:", x[1].shape, "label:", label[1])
    abc = label[1].numpy()
    print(abc)

    backdoor1 = torch.zeros(10, 3, 32, 32)
    for i in range(10):
        aa = torch.load("./backdoor/bdlab2{}.pt".format(i))
        backdoor1[i] = aa[0]

    # model = Lenet5().to(device)
    model = torch.load("model/resnet0050.pt")
    model.linear = torch.nn.AdaptiveAvgPool2d(1)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(model)
    """分类保存10类数据到cifardata，大小为(10,5000,3,32,32)
    for epoch in range(1):
        for batchidx, (x, label) in enumerate(cifar_train):
            print(batchidx)
            if batchidx < 390:
                for i in range(128):
                    la = label[i].numpy()
                    co = count1[la].numpy()
                    cifardata[la,co] = x[i]
                    count1[la] = count1[la] + 1
            if batchidx == 390:
                for i in range(80):
                    la = label[i].numpy()
                    co = count1[la].numpy()
                    cifardata[la,co] = x[i]
                    count1[la] = count1[la] + 1
    print(count1)
    print(cifardata[1,1].shape)
    torch.save(cifardata,'./data/cifar10.pt')
    """

    for epoch in range(200):
        total_num1 = 0
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]

            print(batchidx)
            x, label = x.to(device), label.to(device)
            for i in range(60):
                lab = i % 10
                bd = backdoor1[lab]
                # bd = torch.load('./backdoor/bdlab2{}.pt'.format(lab))
                x[i, :, 23:30, 23:30] = bd[:, 23:30, 23:30]
                label[i] = lab
            pos = buildpos(x, label)
            neg, labneg = buildneg(x, label)

            pos, neg, labneg = pos.to(device), neg.to(device), labneg.to(device)

            total_num1 += x.size(0)

            anchor = model(x).reshape(-1, 2048)
            positive = model(pos).reshape(-1, 2048)
            negetive = model(neg).reshape(-1, 2048)

            loss = criterion(anchor, positive, negetive)
            # loss = tripletloss(anchor,positive,negetive,10,3)
            print(loss)

            # backprop
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        print(epoch, "loss:", loss.item(), "totalnum1:", total_num1)

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, "test acc:", acc, "total：", total_num)
    torch.save(model, "./model/backdoor0050_20_200.pt")


if __name__ == "__main__":
    main()
