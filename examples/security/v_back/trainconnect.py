import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt2
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random

from torch.autograd import Variable

from resnet50 import ResNet50
from resnet34 import ResNet34

from imageio import imsave

# criterion = nn.TripletMarginLoss(margin=10,p=2,reduce=True)
criterion = nn.CrossEntropyLoss()


# mse_loss = torch.nn.MSELoss(reduce=False, size_average=False)
def imshow(img):
    # img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imsave(img, filename):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0)) * 255  # 将像素值映射到 0-255 范围
    image = Image.fromarray(npimg.astype("uint8"))
    image.save(filename)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cpu")

"""
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
"""


# 利用triplet loss训练后门模型
def main():
    batchsz = 50

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
    # model = torch.load('model/backdoor0050_10_3_100.pt')
    model = torch.load("model/backdoor0050_20_200.pt")
    # model.linear = nn.Conv2d(4 * 512, 10, kernel_size=1)
    model.linear = nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), nn.Conv2d(4 * 512, 10, kernel_size=1)
    )
    model = model.to(device)

    optimizer = optim.Adam(model.linear.parameters(), lr=1e-3)
    gradient_step = 1e-3
    # print(model)

    for epoch in range(100):
        total_num1 = 0
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]

            x, label = x.to(device), label.to(device)

            total_num1 += x.size(0)

            anchor = model(x)
            # print(anchor.shape)
            anchor = anchor.reshape(-1, 10)
            # print(anchor.shape)

            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criterion(anchor, label)
            # a = loss.numpy().detach().clone().to('cpu')
            # print(a)
            if loss < 0.001:
                print(loss)
                break
            if batchidx % 30 == 0:
                print(batchidx)
                print(loss)

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

                for i in range(50):
                    lab = i % 10
                    bd = backdoor1[lab]
                    # bd = torch.load('./backdoor/bdlab2{}.pt'.format(lab))

                    x[i, :, 23:30, 23:30] = bd[:, 23:30, 23:30]
                    # x[i, :, 22:29, 22:29] = bd[:, 23:30, 23:30]
                    label[i] = lab

                # [b, 10]
                logits = model(x).reshape(-1, 10)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, "test acc:", acc, "total：", total_num)
        if loss < 0.1:
            print(loss)
            break
    torch.save(model, "./model/backdoor0050.pt")


if __name__ == "__main__":
    main()
