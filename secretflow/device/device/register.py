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
from typing import Callable


# NOTE: Device Conversion Table
# +-------------------+---------------+---------+-------------+
# |     |     PYU     |       SPU     |  TEEU   |      HEU    |
# +-----+-------------+---------------+---------+-------------+
# | PYU |             |      share    | encrypt |    encrypt  |
# +-----+-------------+---------------+---------+-------------+
# | SPU | reconstruct |               |    x    | encrypt+add |
# +-----+-------------+---------------+---------+-------------+
# | TEE |   decrypt   |       x       |         |      x      |
# +-----+-------------+---------------+---------+-------------+
# | HEU |   decrypt   | minus+decrypt |    x    |             |
# +-----+-------------+---------------+---------+-------------+
class DeviceType(IntEnum):
    PYU = 0  # Python Unit
    SPU = 1  # Privacy Preserving Processing Unit
    TEEU = 2  # Trusted Execution Environment Processing Unit
    HEU = 3  # Homomorphic Encryption Unit
    NUM = 4  # Number of device type


class Registrar:
    """Device kernel registry"""

    def __init__(self) -> None:
        self._ops = [{} for _ in range(DeviceType.NUM)]

    def register(self, device_type: DeviceType, name: str, op: Callable):
        """Register device kernel.

        Args:
            device_type (DeviceType): Device type.
            name (str): Op kernel name.
            op (Callable): Op kernel implementaion.

        Raises:
            KeyError: Duplicate device kernel registered.
        """
        if name is None:
            name = op.__name__

        if name in self._ops[device_type]:
            raise KeyError(
                f'device: {device_type}, op: {name} has already been registered'
            )
        self._ops[device_type][name] = op

    def dispatch(self, device_type: DeviceType, name: str, *args, **kwargs):
        """Dispatch device kernel.

        Args:
            device_type (DeviceType): Device type.
            name (str): Op kernel name.

        Raises:
            KeyError: Device Kernel not registered.

        Returns:
            Kernel execution result.
        """
        if name not in self._ops[device_type]:
            raise KeyError(f'device: {device_type}, op: {name} not registered')
        return self._ops[device_type][name](*args, **kwargs)


_registrar = Registrar()


def register(device_type: DeviceType, op_name: str = None):
    """Register device kernel

    Args:
        device_type (DeviceType): Device type.
        op_name (str, optional): Op kernel name. Defaults to None.
    """

    def wrapper(op):
        _registrar.register(device_type, op_name, op)
        return op

    return wrapper


def dispatch(name: str, self, *args, **kwargs):
    """Dispatch device kernel.

    Args:
        name (str): Kernel name.
        self (Device): Traget deivce.

    Returns:
        Kernel execution result.
    """
    return _registrar.dispatch(self.device_type, name, self, *args, **kwargs)
