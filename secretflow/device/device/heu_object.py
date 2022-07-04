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
from heu import phe

from .base import DeviceObject
from .register import dispatch


class HeuInstanceCollection:
    """Heu instance list

    A global variable that stores the Heu instance so that numpy he_ciphertext ndarray
    operations can access heu context information.

    Note: The contents of the HeuInstanceCollection variable are not duplicated between
    Ray Actors. For a specific Actor, the number of meta elements is always 1
    """

    meta = dict()  # heu_instance_id -> {encoder, phe.DestinationHeKit}


class HeCiphertext:
    def __init__(self, cipher: phe.Ciphertext, heu_id):
        self.ct = cipher
        self.heu_id = heu_id
        self.evaluator = HeuInstanceCollection.meta[heu_id].evaluator

    def __getstate__(self):
        return self.ct, self.heu_id

    def __setstate__(self, state):
        self.__init__(*state)

    def __str__(self):
        return f'{HeuInstanceCollection.meta[self.heu_id]}, Ciphertext: {self.ct}'

    def __add__(self, other):
        if isinstance(other, HeCiphertext):
            return HeCiphertext(self.evaluator.add(self.ct, other.ct), self.heu_id)
        elif isinstance(other, phe.Plaintext):
            return HeCiphertext(self.evaluator.add(self.ct, other), self.heu_id)
        elif isinstance(other, (int, float)):
            return HeCiphertext(
                self.evaluator.add(
                    self.ct,
                    HeuInstanceCollection.meta[self.heu_id].encoder.encode(other),
                ),
                self.heu_id,
            )
        else:
            return NotImplementedError(
                f'HeCiphertext + {type(other)} not supported now'
            )

    def __radd__(self, other):
        if isinstance(other, HeCiphertext):
            return HeCiphertext(self.evaluator.add(other.ct, self.ct), self.heu_id)
        elif isinstance(other, phe.Plaintext):
            return HeCiphertext(self.evaluator.add(other, self.ct), self.heu_id)
        elif isinstance(other, (int, float)):
            return HeCiphertext(
                self.evaluator.add(
                    HeuInstanceCollection.meta[self.heu_id].encoder.encode(other),
                    self.ct,
                ),
                self.heu_id,
            )
        else:
            return NotImplementedError(
                f'{type(other)} + HeCiphertext not supported now'
            )

    def __iadd__(self, other):
        if isinstance(other, HeCiphertext):
            self.evaluator.add_inplace(self.ct, other.ct)
        elif isinstance(other, phe.Plaintext):
            self.evaluator.add_inplace(self.ct, other)
        elif isinstance(other, int):
            self.evaluator.add_inplace(
                self.ct, HeuInstanceCollection.meta[self.heu_id].encoder.encode(other)
            )
        else:
            return NotImplementedError(
                f'HeCiphertext += {type(other)} not supported now'
            )
        return self

    def __sub__(self, other):
        if isinstance(other, HeCiphertext):
            return HeCiphertext(self.evaluator.sub(self.ct, other.ct), self.heu_id)
        elif isinstance(other, phe.Plaintext):
            return HeCiphertext(self.evaluator.sub(self.ct, other), self.heu_id)
        elif isinstance(other, (int, float)):
            return HeCiphertext(
                self.evaluator.sub(
                    self.ct,
                    HeuInstanceCollection.meta[self.heu_id].encoder.encode(other),
                ),
                self.heu_id,
            )
        else:
            return NotImplementedError(
                f'HeCiphertext - {type(other)} not supported now'
            )

    def __rsub__(self, other):
        if isinstance(other, HeCiphertext):
            return HeCiphertext(self.evaluator.sub(other.ct, self.ct), self.heu_id)
        elif isinstance(other, phe.Plaintext):
            return HeCiphertext(self.evaluator.sub(other, self.ct), self.heu_id)
        elif isinstance(other, (int, float)):
            return HeCiphertext(
                self.evaluator.sub(
                    HeuInstanceCollection.meta[self.heu_id].encoder.encode(other),
                    self.ct,
                ),
                self.heu_id,
            )
        else:
            return NotImplementedError(
                f'{type(other)} - HeCiphertext not supported now'
            )

    def __isub__(self, other):
        if isinstance(other, HeCiphertext):
            self.evaluator.sub_inplace(self.ct, other.ct)
        elif isinstance(other, phe.Plaintext):
            self.evaluator.sub_inplace(self.ct, other)
        elif isinstance(other, (int, float)):
            self.evaluator.sub_inplace(
                self.ct, HeuInstanceCollection.meta[self.heu_id].encoder.encode(other)
            )
        else:
            return NotImplementedError(
                f'HeCiphertext -= {type(other)} not supported now'
            )
        return self

    def __mul__(self, other):
        if isinstance(other, int):
            return HeCiphertext(self.evaluator.mul(self.ct, other), self.heu_id)
        else:
            return NotImplementedError(
                f'HeCiphertext * {type(other)} not supported now'
            )

    def __rmul__(self, other):
        if isinstance(other, int):
            return HeCiphertext(self.evaluator.mul(other, self.ct), self.heu_id)
        else:
            return NotImplementedError(
                f'{type(other)} * HeCiphertext not supported now'
            )

    def __imul__(self, other):
        if isinstance(other, int):
            self.evaluator.mul_inplace(self.ct, other)
        else:
            return NotImplementedError(
                f'HeCiphertext *= {type(other)} not supported now'
            )
        return self


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
        location_party,
        is_plain: bool = False,
    ):
        super().__init__(device)
        self.data = data
        self.is_plain = is_plain
        self.location = location_party

    def __str__(self):
        return f'is_plain:{self.is_plain}, {self.data}'

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
        return HEUObject(
            self.device,
            self.device.get_participant(self.location).getitem.remote(self.data, item),
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

    def sum(self, axis=None, *args, **kwargs):
        """
        Sum of HeObject elements over a given axis.

        Args:
            axis: None or int or tuple of ints, optional.
                Axis or axes along which a sum is performed. The default, axis=None,
                will sum all the elements of the input array. If axis is negative
                it counts from the last to the first axis.

        Returns:
            sum_along_axis
        """
        return HEUObject(
            self.device,
            self.device.get_participant(self.location).sum.remote(
                self.data, axis, *args, **kwargs
            ),
            self.location,
            self.is_plain,
        )

    def dump(self, path):
        """Dump public key & ciphertext into files."""
        self.device.get_participant(self.location).dump.remote(self.data, path)
