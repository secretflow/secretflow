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

from typing import List, Union
import jax.tree_util
import ray
import numpy as np

from secretflow.device.device.pyu import PYUObject

from .base import DeviceObject
from .register import dispatch


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

    def select_sum(self, item):
        """
        Sum of HEUObject selected elements
        """
        item = jax.tree_util.tree_map(
            lambda x: x.data if isinstance(x, PYUObject) else x, item
        )
        return HEUObject(
            self.device,
            self.device.get_participant(self.location).select_sum.remote(
                self.data, item
            ),
            self.location,
            self.is_plain,
        )

    def batch_select_sum(self, item):
        """
        Sum of HEUObject selected elements
        """
        item = jax.tree_util.tree_map(
            lambda x: x.data if isinstance(x, PYUObject) else x, item
        )

        return HEUObject(
            self.device,
            self.device.get_participant(self.location).batch_select_sum.remote(
                self.data, item
            ),
            self.location,
            self.is_plain,
        )

    def feature_wise_bucket_sum(
        self, subgroup_map, order_map, bucket_num, cumsum=False
    ):
        """
        Sum of HEUObject selected elements
        """

        def process_data(x):
            res = x
            if isinstance(x, PYUObject):
                res = x.data
            return res

        subgroup_map = jax.tree_util.tree_map(process_data, subgroup_map)
        order_map = jax.tree_util.tree_map(process_data, order_map)
        bucket_num = process_data(bucket_num)
        return HEUObject(
            self.device,
            self.device.get_participant(self.location).feature_wise_bucket_sum.remote(
                self.data, subgroup_map, order_map, bucket_num, cumsum
            ),
            self.location,
            self.is_plain,
        )

    def batch_feature_wise_bucket_sum(
        self,
        subgroup_map: List[Union[PYUObject, np.ndarray]],
        order_map: Union[PYUObject, np.ndarray],
        bucket_num: int,
        cumsum=False,
    ) -> List["HEUObject"]:
        """Calculate a list o bucket sum arrays.
        A bucket sum array has dim 2 and shape (feature_num * bucket_sum, self.col_num).

        Args:
            subgroup_map (List[Union[PYUObject, np.ndarray]]): elements in each subset of elements of self.
            order_map (Union[PYUObject, np.ndarray]): shape (self.row_num, feature_num). map[i,j] = k means element i, feature j is in bucket k.
            bucket_num (int): how many bucket to split each feature into.
            cumsum (bool, optional): whether calculate the cumulative sums or individual sums. Defaults to False.

        Return:
            a list of bucket sum array in HEUObject.
        """

        def process_data(x):
            res = x
            if isinstance(x, PYUObject):
                res = x.data
            return res

        subgroup_map = jax.tree_util.tree_map(process_data, subgroup_map)
        order_map = jax.tree_util.tree_map(process_data, order_map)
        bucket_num = process_data(bucket_num)
        return HEUObject(
            self.device,
            self.device.get_participant(
                self.location
            ).batch_feature_wise_bucket_sum.remote(
                self.data, subgroup_map, order_map, bucket_num, cumsum
            ),
            self.location,
            self.is_plain,
        )
