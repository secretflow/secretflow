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

import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def try_gpu(e):
    """Send given tensor to gpu if it is available

    Args:
        e: (torch.Tensor)

    Returns:
        e: (torch.Tensor)
    """
    if torch.cuda.is_available():
        return e.cuda()
    return e


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class RoundDecimal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_digits):
        ctx.save_for_backward(input)
        ctx.n_digits = n_digits
        return torch.round(input * 10**n_digits) / (10**n_digits)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return torch.round(grad_input * 10**ctx.n_digits) / (10**ctx.n_digits), None


torch_round_x_decimal = RoundDecimal.apply


class NumpyDataset(Dataset):
    """This class allows you to convert numpy.array to torch.Dataset

    Args:
        x (np.array):
        y (np.array):
        transform (torch.transform):

    Attriutes
        x (np.array):
        y (np.array):
        transform (torch.transform):
    """

    def __init__(self, x, y=None, transform=None, return_idx=False):
        self.x = x
        self.y = y
        self.transform = transform
        self.return_idx = return_idx

    def __getitem__(self, index):
        x = self.x[index]
        if self.y is not None:
            y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        if not self.return_idx:
            if self.y is not None:
                return x, y
            else:
                return x
        else:
            if self.y is not None:
                return index, x, y
            else:
                return index, x

    def __len__(self):
        """get the number of rows of self.x"""
        return len(self.x)
