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
import ray
import jax.tree_util
from .base import DeviceObject
from .register import dispatch
from secretflow.device.device.pyu import PYUObject


class HEUObject(DeviceObject):
    """HEU Object

    Attributes:
        data: The data hold by this Heu object
        location: The party where the data actually resides
        is_plain: Is the data encrypted or not
    """

    def __init__(
        self,
        device,
        data: ray.ObjectRef,
        location_party: str,
        is_plain: bool = False,
    ):
        super().__init__(device)
        self.data = data
        self.is_plain = is_plain
        assert device.has_party(
            location_party
        ), f"{location_party} is not a party of HEU {id(device)}"
        self.location = location_party

    def __str__(self):
        return f'is_plain:{self.is_plain}, location:{self.location}, {self.data}'

    def __add__(self, other):
        return dispatch('add', self, other)

    def __sub__(self, other):
        return dispatch('sub', self, other)

    def __mul__(self, other):
        return dispatch('mul', self, other)

    def __matmul__(self, other):
        return dispatch('matmul', self, other)

    def __rmatmul__(self, other):
        return dispatch('matmul', self, other)

    def __getitem__(self, item):
        item = jax.tree_util.tree_map(
            lambda x: x.data if isinstance(x, PYUObject) else x, item
        )

        return HEUObject(
            self.device,
            self.device.get_participant(self.location).getitem.remote(self.data, item),
            self.location,
            self.is_plain,
        )

    def __setitem__(self, key, value):
        if isinstance(key, PYUObject):
            key = key.data

        if isinstance(value, HEUObject):
            value = value.data

        return HEUObject(
            self.device,
            self.device.get_participant(self.location).setitem.remote(
                self.data, key, value
            ),
            self.location,
            self.is_plain,
        )

    def encrypt(self, heu_audit_log: str = None):
        """Force encrypt if data is plaintext"""
        if self.is_plain:
            return HEUObject(
                self.device,
                self.device.get_participant(self.location).encrypt.remote(
                    self.data, heu_audit_log
                ),
                self.location,
                False,
            )
        else:
            return self

    def sum(self):
        """
        Sum of HeObject elements over a given axis.

        Returns:
            sum_along_axis
        """
        return HEUObject(
            self.device,
            self.device.get_participant(self.location).sum.remote(self.data),
            self.location,
            self.is_plain,
        )

    def dump(self, path):
        """Dump ciphertext into files."""
        self.device.get_participant(self.location).dump.remote(self.data, path)
