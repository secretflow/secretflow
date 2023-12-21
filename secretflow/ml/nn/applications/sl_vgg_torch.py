from typing import Dict, List, Optional, Union, cast

import torch
import torch.nn as nn
from torch import Tensor

from secretflow.ml.nn.utils import BaseModule

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
    cfg: List[Union[str, int]], input_channels: int = 3, batch_norm: bool = False
) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(input_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            input_channels = v
    return nn.Sequential(*layers)


class VGGBase(BaseModule):
    def __init__(
        self,
        input_channels: int = 3,
        features: Optional[nn.Module] = None,
        init_weights: bool = True,
        classifier=None,
    ) -> None:
        super().__init__()
        self.features = (
            features
            if features is not None
            else make_layers(
                cfgs["D_Mini"], input_channels=input_channels, batch_norm=False
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
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        dnn_units_size=[512 * 3 * 3 * 2, 4096, 4096],
        dnn_activation="relu",
        use_dropout=True,
        dropout: float = 0.5,
        classifier=None,
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
            layers.append(nn.Linear(dnn_units_size[-1], num_classes))
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
