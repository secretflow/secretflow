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

    def to(
        self,
        device: Device,
        spu_vis: str = 'secret',
        heu_dest_party: str = 'auto',
        heu_audit_log: str = None,
    ):
        """Device object conversion.

        Args:
            device (Device): Target device
            spu_vis (str): Deivce object visibility, SPU device only.
              secret: Secret sharing with protocol spdz-2k, aby3, etc.
              public: Public sharing, which means data will be replicated to each node.
            heu_dest_party (str): Where the encrypted data is located, HEU only.

        Returns:
            DeviceObject: Target device object.
        """
        return dispatch('to', self, device, spu_vis, heu_dest_party, heu_audit_log)
