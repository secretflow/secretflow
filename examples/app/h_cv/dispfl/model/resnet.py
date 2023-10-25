# MIT License
#
# Copyright (c) 2022 Rong Dai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义带两个卷积路径和一条捷径的残差基本块类
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1
    ):  # 初始化函数，in_planes为输入通道数，planes为输出通道数，步长默认为1
        super(BasicBlock, self).__init__()
        # 定义第一个卷积，默认卷积前后图像大小不变但可修改stride使其变化，通道可能改变
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # 定义第一个批归一化
        self.bn1 = nn.BatchNorm2d(planes)
        # 定义第二个卷积，卷积前后图像大小不变，通道数不变
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # 定义第二个批归一化
        self.bn2 = nn.BatchNorm2d(planes)

        # 定义一条捷径，若两个卷积前后的图像尺寸有变化(stride不为1导致图像大小变化或通道数改变)，捷径通过1×1卷积用stride修改大小
        # 以及用expansion修改通道数，以便于捷径输出和两个卷积的输出尺寸匹配相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    # 定义前向传播函数，输入图像为x，输出图像为out
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积和第一个批归一化后用ReLU函数激活
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 第二个卷积和第二个批归一化后与捷径相加
        out = F.relu(out)  # 两个卷积路径输出与捷径输出相加后用ReLU激活
        return out


# 定义残差网络ResNet18
class ResNet(nn.Module):
    # 定义初始函数，输入参数为残差块，残差块数量，默认参数为分类数10
    def __init__(self, block, num_blocks, class_num=10):
        super(ResNet, self).__init__()
        # 设置第一层的输入通道数
        self.in_planes = 64

        # 定义输入图片先进行一次卷积与批归一化，使图像大小不变，通道数由3变为64得两个操作
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 定义第一层，输入通道数64，有num_blocks[0]个残差块，残差块中第一个卷积步长自定义为1
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # 定义第二层，输入通道数128，有num_blocks[1]个残差块，残差块中第一个卷积步长自定义为2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # 定义第三层，输入通道数256，有num_blocks[2]个残差块，残差块中第一个卷积步长自定义为2
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # 定义第四层，输入通道数512，有num_blocks[3]个残差块，残差块中第一个卷积步长自定义为2
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 定义全连接层，输入512*block.expansion个神经元，输出10个分类神经元
        self.linear = nn.Linear(512 * block.expansion, class_num)

    # 定义创造层的函数，在同一层中通道数相同，输入参数为残差块，通道数，残差块数量，步长
    def _make_layer(self, block, planes, num_blocks, stride):
        # strides列表第一个元素stride表示第一个残差块第一个卷积步长，其余元素表示其他残差块第一个卷积步长为1
        strides = [stride] + [1] * (num_blocks - 1)
        # 创建一个空列表用于放置层
        layers = []
        # 遍历strides列表，对本层不同的残差块设置不同的stride
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 创建残差块添加进本层
            self.in_planes = (
                planes * block.expansion
            )  # 更新本层下一个残差块的输入通道数或本层遍历结束后作为下一层的输入通道数
        return nn.Sequential(*layers)  # 返回层列表

    # 定义前向传播函数，输入图像为x，输出预测数据
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积和第一个批归一化后用ReLU函数激活
        out = self.layer1(out)  # 第一层传播
        out = self.layer2(out)  # 第二层传播
        out = self.layer3(out)  # 第三层传播
        out = self.layer4(out)  # 第四层传播
        out = F.avg_pool2d(out, 4)  # 经过一次4×4的平均池化
        out = out.view(out.size(0), -1)  # 将数据flatten平坦化
        out = self.linear(out)  # 全连接传播
        return out


def customized_resnet18(
    pretrained: bool = False, class_num=10, progress: bool = True
) -> ResNet:
    res18 = ResNet(BasicBlock, [2, 2, 2, 2], class_num=class_num)

    # Change BN to GN
    res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

    res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

    res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

    assert len(dict(res18.named_parameters()).keys()) == len(
        res18.state_dict().keys()
    ), 'More BN layers are there...'

    return res18


class tiny_ResNet(nn.Module):
    # 定义初始函数，输入参数为残差块，残差块数量，默认参数为分类数10
    def __init__(self, block, num_blocks, class_num=10):
        super(tiny_ResNet, self).__init__()
        # 设置第一层的输入通道数
        self.in_planes = 64

        # 定义输入图片先进行一次卷积与批归一化，使图像大小不变，通道数由3变为64得两个操作
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 定义第一层，输入通道数64，有num_blocks[0]个残差块，残差块中第一个卷积步长自定义为1
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # 定义第二层，输入通道数128，有num_blocks[1]个残差块，残差块中第一个卷积步长自定义为2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # 定义第三层，输入通道数256，有num_blocks[2]个残差块，残差块中第一个卷积步长自定义为2
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # 定义第四层，输入通道数512，有num_blocks[3]个残差块，残差块中第一个卷积步长自定义为2
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 定义全连接层，输入512*block.expansion个神经元，输出10个分类神经元
        self.linear = nn.Linear(512 * block.expansion, class_num)

    # 定义创造层的函数，在同一层中通道数相同，输入参数为残差块，通道数，残差块数量，步长
    def _make_layer(self, block, planes, num_blocks, stride):
        # strides列表第一个元素stride表示第一个残差块第一个卷积步长，其余元素表示其他残差块第一个卷积步长为1
        strides = [stride] + [1] * (num_blocks - 1)
        # 创建一个空列表用于放置层
        layers = []
        # 遍历strides列表，对本层不同的残差块设置不同的stride
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 创建残差块添加进本层
            self.in_planes = (
                planes * block.expansion
            )  # 更新本层下一个残差块的输入通道数或本层遍历结束后作为下一层的输入通道数
        return nn.Sequential(*layers)  # 返回层列表

    # 定义前向传播函数，输入图像为x，输出预测数据
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积和第一个批归一化后用ReLU函数激活
        out = self.layer1(out)  # 第一层传播
        out = self.layer2(out)  # 第二层传播
        out = self.layer3(out)  # 第三层传播
        out = self.layer4(out)  # 第四层传播
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)  # 全连接传播
        return out


def tiny_resnet18(
    pretrained: bool = False, class_num=10, progress: bool = True
) -> ResNet:
    res18 = tiny_ResNet(BasicBlock, [2, 2, 2, 2], class_num=class_num)

    # Change BN to GN
    res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

    res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

    res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

    assert len(dict(res18.named_parameters()).keys()) == len(
        res18.state_dict().keys()
    ), 'More BN layers are there...'

    return res18


# def resnet18(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, num_classes=10,**kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=32, num_channels=group_channels)
#
#     model = ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)
#     return model


# def resnet34(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(BasicBlock, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet34'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnet50(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet50'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnet101(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet101'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnet152(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(Bottleneck, [3, 8, 36, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet152'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnext50_32x4d(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                     **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnext101_32x8d(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                      **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def wide_resnet50_2(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                     **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['width_per_group'] = 64 * 2
#     model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['wide_resnet50_2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def wide_resnet101_2(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                      **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['width_per_group'] = 64 * 2
#     model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['wide_resnet101_2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
