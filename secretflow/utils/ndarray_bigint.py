# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random

import numpy as np
from heu import numpy as hnp


def randbits(shape: tuple, bits):
    items = math.prod(shape)
    data = [random.getrandbits(bits) - (1 << (bits - 1)) for _ in range(items)]
    return BigintNdArray(data, shape)


def randint(shape: tuple, min, max):
    items = math.prod(shape)
    data = [random.randint(min, max) for _ in range(items)]
    return BigintNdArray(data, shape)


def arange(max):
    return BigintNdArray(list(range(max)), (max,))


def zeros(shape):
    return BigintNdArray([0] * math.prod(shape), shape)


class BigintNdArray:
    def __init__(self, data, shape):
        assert len(data) == math.prod(shape), f"{len(data)} != {math.prod(shape)}"
        self.shape = shape
        self.data = data

    def resize(self, shape):
        assert math.prod(shape) == math.prod(
            self.shape
        ), f"cannot resize array of size {self.shape} into shape {shape}"
        self.shape = shape

    def __to_list(self, dim, idx):  # idx is a list just to make it pass by ref
        if dim == len(self.shape) - 1:
            dim_res = self.data[idx[0] : idx[0] + self.shape[dim]]
            idx[0] += self.shape[dim]
            return dim_res
        else:
            return [self.__to_list(dim + 1, idx) for _ in range(self.shape[dim])]

    def to_list(self):
        return self.__to_list(0, [0])

    def to_numpy(self):
        return np.array(self.to_list())

    def to_hnp(self, encoder):
        return hnp.array(self.to_list(), encoder=encoder)

    def to_bytes(self, bytes_per_int, byteorder='little'):
        mask = (1 << bytes_per_int * 8) - 1
        res = bytearray()
        for d in self.data:
            res += (d & mask).to_bytes(bytes_per_int, byteorder)
        return bytes(res)

    def __str__(self):
        return str(self.to_list())

    def __add__(self, other):
        assert (
            self.shape == other.shape
        ), f"Int128 arrays do not support broadcasting, their shape must be the same"
        return BigintNdArray([a + b for a, b in zip(self.data, other.data)], self.shape)

    def __iadd__(self, other):
        assert (
            self.shape == other.shape
        ), f"Int128 arrays do not support broadcasting, their shape must be the same"
        self.data = [a + b for a, b in zip(self.data, other.data)]
