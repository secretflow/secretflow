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

from enum import IntEnum


# NOTE: Device Conversion Table
# +-------------------+---------------+---------+-------------+
# |     |     PYU     |       PPU     |  TEE    |      HE     |
# +-----+-------------+---------------+---------+-------------+
# | PYU |             |      share    | encrypt |    encrypt  |
# +-----+-------------+---------------+---------+-------------+
# | PPU | reconstruct |               |    x    | encrypt+add |
# +-----+-------------+---------------+---------+-------------+
# | TEE |   decrypt   |       x       |         |      x      |
# +-----+-------------+---------------+---------+-------------+
# | HE  |   decrypt   | minus+decrypt |    x    |             |
# +-----+-------------+---------------+---------+-------------+
class DeviceType(IntEnum):
    PYU = 0  # Python Unit
    PPU = 1  # Privacy Preserving Processing Unit
    TEE = 2  # Trusted Execution Environment
    HEU = 3  # Homomorphic Encryption Unit
    NUM = 4  # Number of device type


class Registrar:
    def __init__(self) -> None:
        self._ops = [{} for _ in range(DeviceType.NUM)]

    def register(self, device_type, name, op):
        if name is None:
            name = op.__name__

        if name in self._ops[device_type]:
            raise KeyError(
                f'device: {device_type}, op: {name} has already been registered')
        self._ops[device_type][name] = op

    def dispatch(self, device_type, name, *args, **kwargs):
        if name not in self._ops[device_type]:
            raise KeyError(f'device: {device_type}, op: {name} not registered')
        return self._ops[device_type][name](*args, **kwargs)


_registrar = Registrar()


def register(device_type, op_name=None):
    """注册Object kernel"""

    def wrapper(op):
        _registrar.register(device_type, op_name, op)
        return op

    return wrapper


def dispatch(name, self, *args, **kwargs):
    """分发Object kernel"""
    # TODO(@xibin.wxb): 当args是混合类型时，根据哪个arg决定device_type?
    return _registrar.dispatch(self.device_type, name, self, *args, **kwargs)
