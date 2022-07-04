from typing import Optional

import torch
from filelock import FileLock
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from secretflow.ml.nn.fl_base import BaseModule
from secretflow.ml.nn.sl_base import SLBaseModule


class ConvNet(BaseModule):
    """Small ConvNet for MNIST."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        self.fc = nn.Linear(self.fc_in_dim, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.fc_in_dim)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# TODO: check https://blog.openmined.org/split-neural-networks-on-pysyft/ for a more unified SL model implementation
# relies on `model.send(location)` function


class SLConvNet(SLBaseModule):
    """Split Learning模型."""

    def __init__(
        self,
        output_shape: int,
        fc_in_dim: int,
        embed_shape_from_other: Optional[int] = None,
        with_softmax=False,
    ):
        super(SLConvNet, self).__init__()

        self.with_softmax = with_softmax

        self.fc_in_dim = fc_in_dim
        self.embed_shape_from_other = embed_shape_from_other

        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)

        # 若有传入embedding, concat后再forward
        if embed_shape_from_other is not None:
            self.fc = nn.Linear(self.fc_in_dim + embed_shape_from_other, output_shape)
        else:
            self.fc = nn.Linear(self.fc_in_dim, output_shape)

        self.out_tensor = None

    def forward(self, x, embed_from_other=None):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.fc_in_dim)

        if embed_from_other is not None:
            self.embed_from_other: torch.Tensor = embed_from_other
            x = torch.cat([x, self.embed_from_other], dim=1)

        x = self.fc(x)

        if self.with_softmax:
            self.out_tensor = F.log_softmax(x, dim=1)
        else:
            self.out_tensor = x
        return self.out_tensor

    def return_embed_grad(self):
        return self.embed_from_other.grad

    def backward_embed_grad(self, gradients):
        self.out_tensor.backward(gradients)


def get_mnist_dataloader(
    is_train=True, loc='./data', num_batch_size=128, use_cols=None, with_y=True
):
    with FileLock("./data.lock"):
        mnist = datasets.MNIST(
            loc,
            train=is_train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
    if use_cols is not None:
        mnist.data = mnist.data[:, :, use_cols]
    if with_y is not True:
        mnist.targets = [-1 for i in range(mnist.targets.size()[0])]

    dataloader = DataLoader(mnist, batch_size=num_batch_size, shuffle=True)
    return dataloader


def calculate_accu(y_pred: torch.Tensor, y: torch.Tensor):
    pred_label = torch.argmax(y_pred, dim=-1)
    accu = torch.sum(pred_label == y)
    return accu / len(y_pred)
