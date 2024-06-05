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

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from secretflow.ml.nn.sl.defenses.fed_pass import ConvPassportBlock, LinearPassportBlock


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        preprocess_layer: Optional[Callable[..., nn.Module]] = None,
        use_passport: bool = False,
    ) -> None:
        super().__init__()
        self.preprocess_layer = preprocess_layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if not use_passport:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = ConvPassportBlock(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if self.preprocess_layer is not None:
            x = self.preprocess_layer(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):
    def __init__(
        self,
        block: type(BasicBlock),
        layers: List[int],
        input_channels: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        use_passport: bool = False,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        classifier: Optional[nn.Module] = None,
        preprocess_layer=None,
    ) -> None:
        super().__init__()
        self.preprocess_layer = preprocess_layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.classifier = classifier

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            input_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            use_passport=use_passport,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # for BatchNorm2d, affine=False module has no weight/bias
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        use_passport: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            passport = False if i != blocks - 1 else use_passport

            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_passport=passport,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        if self.preprocess_layer:
            x = self.preprocess_layer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.classifier is not None:
            x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def output_num(self):
        return 1


class ResNetFuse(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        dnn_units_size: List[int] = [512 * 2],
        dnn_activation: Optional[str] = "relu",
        use_dropout: bool = False,
        dropout: float = 0.5,
        use_passport: bool = False,
        classifier: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super(ResNetFuse, self).__init__()
        if classifier is not None:
            self.classifier = classifier
        else:
            layers = []
            for i in range(1, len(dnn_units_size)):
                layers.append(nn.Linear(dnn_units_size[i - 1], dnn_units_size[i]))
                if dnn_activation is not None and dnn_activation == 'relu':
                    layers.append(nn.ReLU(True))
                if use_dropout:
                    layers.append(nn.Dropout(p=dropout))

            if not use_passport:
                layers.append(nn.Linear(dnn_units_size[-1], num_classes))
            else:
                layers.append(
                    LinearPassportBlock(
                        dnn_units_size[-1],
                        num_classes,
                        hidden_feature=kwargs.get('passport_hidden_size', 32),
                        num_passport=kwargs.get('passport_num_passport', 1),
                    )
                )

            self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        fuse_input = torch.cat(inputs, dim=1)
        outputs = self.classifier(fuse_input)
        return outputs


# just for exp
class NaiveSumSoftmax(nn.Module):
    def __init__(self, use_softmax=True):
        super().__init__()
        self.linear = nn.Linear(10, 10)  # just to pass optimizer in SLModel
        self.use_softmax = use_softmax
        if use_softmax:
            self.layer = nn.Softmax(dim=-1)

    def forward(self, x: List[Tensor]) -> Tensor:
        out = x[0] + x[1]
        if self.use_softmax:
            out = self.layer(out)
        return out
