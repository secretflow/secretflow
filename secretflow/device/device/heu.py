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
from phe import paillier

from .base import Device, DeviceType, DeviceObject
from .register import dispatch


class HEUObject(DeviceObject):
    """同态加密Object

    Attributes:
        data: 加密数据
    """

    def __init__(self, device: Device, data: ray.ObjectRef):
        super().__init__(device)
        self.data = data

    def __add__(self, other):
        return dispatch('add', self, other)

    def __sub__(self, other):
        return dispatch('sub', self, other)

    def __mul__(self, other):
        return dispatch('mul', self, other)

    def __matmul__(self, other):
        return dispatch('matmul', self, other)


class HEU(Device):
    """同态加密设备
    """

    def __init__(self, config):
        """初始化同态加密设备

        Args:
            config: 同态加密配置，示例：
            ```python
            {
                'generator': 'alice',
                'evaluator': 'bob',
                'key_size': 2048,
            }
            ```
        """
        super().__init__(DeviceType.HEU)

        self.generator = config['generator']
        self.evaluator = config['evaluator']

        self.pk, self.sk = self.generate_keypair.options(
            resources={self.generator: 1}).remote(config['key_size'])

    def __call__(self, fn):
        raise NotImplementedError()

    @classmethod
    @ray.remote(num_returns=2)
    def generate_keypair(key_size):
        return paillier.generate_paillier_keypair()

    @classmethod
    @ray.remote
    def encrypt(pk, data):
        return np.array([pk.encrypt(x) for x in data.flatten()]).reshape(data.shape)

    @classmethod
    @ray.remote
    def decrypt(sk, data):
        return np.array([sk.decrypt(x) for x in data.flatten()]).reshape(data.shape)

    @classmethod
    @ray.remote
    def a2h(s1, s2):
        return np.add(s1, s2)
