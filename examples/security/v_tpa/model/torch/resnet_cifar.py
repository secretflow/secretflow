#!/usr/bin/env python
# coding=utf-8
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

import sys

sys.path.append("..")

import pdb

import torch.nn as nn
import torch.nn.functional as F
from tools.dp_utils import NaiveLayer
from torchsummary import summary


class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out = out + residual
        return out


class ResNetCIFAR10(nn.Module):
    """
    A Residual network.
    """

    def __init__(self, n_class):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=False),
            ResidualBlock(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.Linear(in_features=128, out_features=n_class),
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


split_options = [3, 6, 11, 18]
dist_weights = [0.35, 0.35, 0.50, 1.5]


class Splittable_ResNetCIFAR10:
    """
    A Residual network.
    """

    def __init__(self, n_class):
        self.layers = [
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=False),
            ResidualBlock(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=128),
            nn.Linear(in_features=128, out_features=n_class),
        ]

    def split(self, split_index, party_num):
        if split_index in [3, 6]:
            bottom_model = nn.Sequential(*self.layers[:split_index], NaiveLayer())
        else:
            bottom_model = nn.Sequential(*self.layers[:split_index])

        layers = self.layers[split_index:]
        if split_index == 18:
            layers[0] = nn.Linear(128 * party_num, 128)
        fuse_model = nn.Sequential(*layers)
        return bottom_model, fuse_model


class Splittable_SimpleCIFAR10:
    pass


class SimpleCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avg_pool(out)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    # model = ResNetCIFAR10()
    # split_cifar10 = Splittable_ResNetCIFAR10()
    # bottom, fuse_model = split_cifar10.split(5)
    model = SimpleCIFAR10(10)
    pdb.set_trace()
    print("end")
