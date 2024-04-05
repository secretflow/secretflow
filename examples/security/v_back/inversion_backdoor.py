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
from torch.utils.data import DataLoader
from torchvision import datasets

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random

from imageio import imsave
from resnet34 import ResNet34
from resnet50 import ResNet50
from torch.autograd import Variable

criterion = nn.CrossEntropyLoss()


# mse_loss = torch.nn.MSELoss(reduce=False, size_average=False)
def imshow(img):
    # img = img / 2 + 0.5
    npimg = img.numpy()
    """
    img 格式： channels,imageSize,imageSize
    imshow需要格式：imageSize,imageSize,channels
    np.transpose 转换数组
    """
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imagesave(img, imgname):
    npimg = img.numpy()
    npimg = npimg.reshape(3, 32, 32)
    npimg = npimg.transpose(1, 2, 0)
    imsave(imgname, npimg)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)
    # model = ResNet34().to(device)
    # model.state_dict = torch.load('model/resnet34.pth')
    model = torch.load("model/resnet0050.pt")

    model.eval()

    transform_validation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    validation_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_validation
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=100, shuffle=True, num_workers=2
    )

    dataIter = iter(validation_loader)
    x, labels = dataIter.next()
    x1 = x[0]
    # imshow(x1)
    x11 = x1.to("cpu")
    x1 = x1.to(device)
    x1 = Variable(torch.unsqueeze(x1, dim=0).float(), requires_grad=False)
    # imshow(x1)
    # for i in range(10):
    #    imagesave(x[i],'./pic/temp{}.png'.format(i))

    labels1 = labels[0]
    labels1 = torch.tensor([labels1]).to(device)
    print(labels1.shape)

    x2 = x[1]
    x2 = x2.to(device)
    x2 = Variable(torch.unsqueeze(x2, dim=0).float(), requires_grad=True)
    # print(x1.shape)

    y = model(x1)
    y = y.reshape(-1, 10)
    # print(y)
    # print(labels[0])

    tensor = torch.zeros(3, 32, 32).to(device)
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=True)
    image = torch.zeros(1, 3, 32, 32).to(device)
    # tensor = x2

    gradient_step = 0.005

    for i in range(200):
        tensor.requires_grad = True

        yhat = model(tensor)
        yhat = yhat.reshape(-1, 10)
        # print(yhat.shape)

        loss = criterion(yhat, labels1)

        loss.backward(retain_graph=True)
        # print(tensor.grad)
        print(loss)

        with torch.no_grad():
            # apply gradient descent
            tensor[:, :, 23:30, 23:30] = (
                tensor[:, :, 23:30, 23:30]
                - gradient_step * tensor.grad[:, :, 23:30, 23:30]
            )

            # tensor = torch.tensor(tensor)

        # with torch.no_grad():
        # print(x2.grad)
        # x2 = (x2 - gradient_step * x2.grad)

    print(model(tensor).reshape(-1, 10))
    print(y)
    image = tensor.squeeze(0).detach().clone().to("cpu")
    print(x11.shape)
    print(image.shape)
    imshow(x11)
    imshow(image)


if __name__ == "__main__":
    main()
