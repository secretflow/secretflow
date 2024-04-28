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

from typing import Callable, Dict, List, Optional, Union, cast

import torch
import torch.nn as nn
from torch import Tensor

from secretflow.ml.nn.core.torch import BaseModule
from secretflow.ml.nn.sl.defenses.fed_pass import ConvPassportBlock, LinearPassportBlock

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "D_Mini": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def make_layers(
    cfg: List[Union[str, int]],
    input_channels: int = 3,
    batch_norm: bool = False,
    use_passport: bool = False,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            if i != len(cfg) - 1 or not use_passport:
                conv2d = nn.Conv2d(input_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = ConvPassportBlock(input_channels, v, 3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            input_channels = v
    return nn.Sequential(*layers)


class VGGBase(BaseModule):
    def __init__(
        self,
        input_channels: int = 3,
        features: Optional[nn.Module] = None,
        init_weights: bool = True,
        use_passport: bool = False,
        classifier: Optional[nn.Module] = None,
        preprocess_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.preprocess_layer = preprocess_layer
        self.features = (
            features
            if features is not None
            else make_layers(
                cfgs["D_Mini"],
                input_channels=input_channels,
                batch_norm=False,
                use_passport=use_passport,
            )
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = classifier

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    # for Linear(bias=False)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preprocess_layer is not None:
            x = self.preprocess_layer(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

    def output_num(self):
        return 1


class VGGFuse(BaseModule):
    def __init__(
        self,
        num_classes: int = 10,
        dnn_units_size: List[int] = [512 * 3 * 3 * 2, 4096, 4096],
        dnn_activation: str = "relu",
        use_dropout: bool = True,
        dropout: float = 0.5,
        use_passport: bool = False,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super(VGGFuse, self).__init__()
        if classifier is not None:
            self.classifier = classifier
        else:
            layers = []
            for i in range(1, len(dnn_units_size)):
                layers.append(nn.Linear(dnn_units_size[i - 1], dnn_units_size[i]))
                if dnn_activation == 'relu':
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


# just for attack exp
class NaiveSumSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)  # just to pass optimizer in SLModel
        self.layer = nn.Softmax(dim=-1)

    def forward(self, x: List[Tensor]) -> Tensor:
        x = x[0] + x[1]
        out = self.layer(x)
        return out
