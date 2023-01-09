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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from heu import phe

from .register import DeviceType, dispatch


class Device(ABC):
    def __init__(self, device_type: DeviceType):
        """Abstraction device base class.

        Args:
            device_type (DeviceType): underlying device type
        """
        self._device_type = device_type

    @property
    def device_type(self):
        """Get underlying device type"""
        return self._device_type

    @abstractmethod
    def __call__(self, fn, **kwargs):
        """Set up ``fn`` for scheduling to this device"""
        pass


@dataclass
class MoveConfig:
    spu_vis: str = 'secret'
    """spu_vis (str): Deivce object visibility, SPU device only. Value can be:
        - secret: Secret sharing with protocol spdz-2k, aby3, etc.
        - public: Public sharing, which means data will be replicated to each node.
    """

    heu_dest_party: str = 'auto'
    """Where the encrypted data is located"""

    heu_encoder: Union[
        phe.IntegerEncoder,
        phe.FloatEncoder,
        phe.BigintEncoder,
        phe.BatchEncoder,
        phe.IntegerEncoderParams,
        phe.FloatEncoderParams,
        phe.BigintEncoderParams,
        phe.BatchEncoderParams,
    ] = None

    """Do encode before move data to heu"""

    heu_audit_log: str = None
    """file path to record audit log"""


class DeviceObject(ABC):
    def __init__(self, device: Device):
        """Abstraction device object base class.

        Args:
            device (Device): Device where this object is located.
        """
        self.device = device

    @property
    def device_type(self):
        """Get underlying device type"""
        return self.device.device_type

    def to(self, device: Device, config: MoveConfig = None):
        """Device object conversion.

        Args:
            device (Device): Target device
            config: configuration of this data movement

        Returns:
            DeviceObject: Target device object.
        """
        assert isinstance(
            config, (type(None), MoveConfig)
        ), f"config must be MoveConfig type, got {type(config)}, value={config}"

        return dispatch(
            'to', self, device, config if config is not None else MoveConfig()
        )
