# src: https://github.com/FedML-AI/FedML/blob/ecd2d81222301d315ca3a84be5a5ce4f33d6181c/python/fedml/model/cv/resnet_gn.py

import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ["ResNet", "resnet18"]

from model.group_normalization import GroupNorm2d

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def norm2d(planes, num_channels_per_group=32):
    # print("num_channels_per_group:{}".format(num_channels_per_group))
    if num_channels_per_group > 0:
        return GroupNorm2d(
            planes, num_channels_per_group, affine=True, track_running_stats=False
        )
    else:
        return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_norm=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm2d(planes, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm2d(planes, group_norm)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_norm=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm2d(planes, group_norm)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm2d(planes * 4, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, group_norm=2):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.prob_policy_net = PolicyNet()

        self.global_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.local_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm2d(64, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.global_layer1 = self._make_layer(block, 64, layers[0], group_norm=group_norm)
        self.global_layer2 = self._make_layer(block, 128, layers[1], stride=2, group_norm=group_norm)
        self.global_layer3 = self._make_layer(block, 256, layers[2], stride=2, group_norm=group_norm)
        self.global_layer4 = self._make_layer(block, 512, layers[3], stride=2, group_norm=group_norm)

        self.inplanes = 64
        self.local_layer1 = self._make_layer(block, 64, layers[0], group_norm=group_norm)
        self.local_layer2 = self._make_layer(block, 128, layers[1], stride=2, group_norm=group_norm)
        self.local_layer3 = self._make_layer(block, 256, layers[2], stride=2, group_norm=group_norm)
        self.local_layer4 = self._make_layer(block, 512, layers[3], stride=2, group_norm=group_norm)

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d(1)

        self.global_fc = nn.Linear(512 * block.expansion, num_classes)
        self.local_fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, GroupNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.fill_(0)
            if isinstance(m, BasicBlock):
                m.bn2.weight.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1, group_norm=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm2d(planes * block.expansion, group_norm),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, group_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_norm=group_norm))

        return nn.Sequential(*layers)

    def forward(self, x, mode='personalized', hard_decision=False):
        # print(x.abs().min(), x.abs().max())

        probs = None

        if mode == 'local':
            x = self.local_conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.local_layer1(x)
            x = self.local_layer2(x)
            x = self.local_layer3(x)
            x = self.local_layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.local_fc(x)

        elif mode == 'global':
            x = self.global_conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.global_layer1(x)
            x = self.global_layer2(x)
            x = self.global_layer3(x)
            x = self.global_layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.global_fc(x)

        elif mode == 'personalized':
            p1, p2, p3, p4, p5, p6 = self.prob_policy_net(x)

            if hard_decision:
                p1, p2, p3, p4, p5, p6 = torch.round(p1), torch.round(p2), torch.round(p3), torch.round(
                    p4), torch.round(p5), torch.round(p6)
                # print(p1.shape, p2.shape, p3.shape, p4.shape, p5.shape, p6.shape)

            global_x = self.global_conv1(x)
            local_x = self.local_conv1(x)

            x = global_x * p1[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) + local_x * p1[:,
                                                                                                           1].unsqueeze(
                1).unsqueeze(2).unsqueeze(3).expand_as(local_x)

            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            global_x = self.global_layer1(x)
            local_x = self.local_layer1(x)
            x = global_x * p2[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) + local_x * p2[:,
                                                                                                           1].unsqueeze(
                1).unsqueeze(2).unsqueeze(3).expand_as(local_x)

            global_x = self.global_layer2(x)
            local_x = self.local_layer2(x)
            x = global_x * p3[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) + local_x * p3[:,
                                                                                                           1].unsqueeze(
                1).unsqueeze(2).unsqueeze(3).expand_as(local_x)

            global_x = self.global_layer3(x)
            local_x = self.local_layer3(x)
            x = global_x * p4[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) + local_x * p4[:,
                                                                                                           1].unsqueeze(
                1).unsqueeze(2).unsqueeze(3).expand_as(local_x)

            global_x = self.global_layer4(x)
            local_x = self.local_layer4(x)
            x = global_x * p5[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) + local_x * p5[:,
                                                                                                           1].unsqueeze(
                1).unsqueeze(2).unsqueeze(3).expand_as(local_x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            global_x = self.global_fc(x)
            local_x = self.local_fc(x)
            x = global_x * p6[:, 0].unsqueeze(1).expand_as(global_x) + local_x * p6[:, 1].unsqueeze(1).expand_as(
                local_x)

            probs = torch.cat(
                [p1[:, :, None], p2[:, :, None], p3[:, :, None], p4[:, :, None], p5[:, :, None], p6[:, :, None]], dim=2)
        return x, probs


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


class PolicyNet(nn.Module):

    def __init__(self) -> None:
        super(PolicyNet, self).__init__()

        self.linear_1 = nn.Linear(3072, 1000)
        self.linear_2 = nn.Linear(1000, 750)
        self.linear_3 = nn.Linear(750, 350)
        self.linear_4 = nn.Linear(350, 150)
        self.linear_5 = nn.Linear(150, 50)
        self.linear_6 = nn.Linear(50, 2)

        self.linear_1_exit = nn.Linear(1000, 2)
        self.linear_2_exit = nn.Linear(750, 2)
        self.linear_3_exit = nn.Linear(350, 2)
        self.linear_4_exit = nn.Linear(150, 2)
        self.linear_5_exit = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x1 = self.linear_1(x)
        p1 = F.softmax(self.linear_1_exit(x1), dim=-1)
        x1 = F.relu(x1)

        x2 = self.linear_2(x1)
        p2 = F.softmax(self.linear_2_exit(x2), dim=-1)
        x2 = F.relu(x2)

        x3 = self.linear_3(x2)
        p3 = F.softmax(self.linear_3_exit(x3), dim=-1)
        x3 = F.relu(x3)

        x4 = self.linear_4(x3)
        p4 = F.softmax(self.linear_4_exit(x4), dim=-1)
        x4 = F.relu(x4)

        x5 = self.linear_5(x4)
        p5 = F.softmax(self.linear_5_exit(x5), dim=-1)
        x5 = F.relu(x5)

        x6 = F.softmax(self.linear_6(x5), dim=-1)

        return p1, p2, p3, p4, p5, x6