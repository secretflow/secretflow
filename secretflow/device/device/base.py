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

from .register import DeviceType, dispatch, register


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


def _name_of_to(device_type: DeviceType):
    return f'to_{device_type.name}'


class DeviceObject(ABC):
    device: 'Device'

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

    def to(self, device: Device, *args, **kwargs):
        """Device object conversion.

        Args:
            device (Device): Target device
            config: configuration of this data movement

        Returns:
            DeviceObject: Target device object.
        """
        return dispatch(_name_of_to(device.device_type), self, device, *args, **kwargs)


def register_to(from_device_type, to_device_type):
    """Register to as device kernel.

    Args:
        from_device_type: the source device type.
        to_device_type: the dest device type.
    """

    return register(device_type=from_device_type, op_name=_name_of_to(to_device_type))
