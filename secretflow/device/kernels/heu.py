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

import numpy as np
import ray

from secretflow.device.device import HEUObject, PYU, PYUObject
from secretflow.device.device import register, DeviceType, Device


@register(DeviceType.HEU)
def to(self, device: Device, vis):
    assert isinstance(device, Device)

    if isinstance(device, PYU):
        assert device.party == self.device.generator, f'Can not convert to PYU device without secret key'
        data = self.device.decrypt.options(resources={self.device.generator: 1}).remote(self.device.sk, self.data)
        return PYUObject(device, data)

    raise ValueError(f'Unexpected device type: {type(device)}')


@ray.remote
def _math_op(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _binary_op(self: HEUObject, other: HEUObject, op) -> 'HEUObject':
    assert isinstance(other, HEUObject)
    data = _math_op.options(resources={self.device.evaluator: 1}).remote(op, self.data, other.data)
    return HEUObject(self.device, data)


@register(DeviceType.HEU)
def add(self, other):
    return _binary_op(self, other, np.add)


@register(DeviceType.HEU)
def sub(self, other):
    return _binary_op(self, other, np.subtract)


@register(DeviceType.HEU)
def mul(self, other):
    return _binary_op(self, other, np.multiply)


@register(DeviceType.HEU)
def matmul(self, other):
    return _binary_op(self, other, np.dot)
