# Copyright 2025 Ant Group Co., Ltd.
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

import dataclasses
from typing import List, Tuple, Union

import jax.numpy as jnp
import numpy as np
import spu

from secretflow.data.ndarray.ndarray import FedNdarray, PartitionWay
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.device.spu import SPU, SPUObject


@dataclasses.dataclass
class BaseModelData:
    core: SPUObject


class _ModelBase:
    def __init__(self, spu: SPU):
        self.spu = spu
        self.enable_spu_cache = (
            hasattr(spu, "experimental")
            and hasattr(getattr(spu, "experimental"), "make_cached_var")
            and hasattr(getattr(spu, "experimental"), "drop_cached_var")
        )

    def _prepare_dataset(
        self, ds: Union[FedNdarray, VDataFrame]
    ) -> Tuple[FedNdarray, Tuple[int, int]]:
        """
        check data setting and get total shape.

        Args:
            ds: input dataset

        Return:
            First: dataset in unified type
            Second: shape concat all partition.
        """
        assert isinstance(
            ds, (FedNdarray, VDataFrame)
        ), f"ds should be FedNdarray or VDataFrame: {type(ds)}"
        ds = ds if isinstance(ds, FedNdarray) else ds.values
        shapes = ds.partition_shape()
        assert len(shapes) > 0, "input dataset is empty"
        assert ds.partition_way == PartitionWay.VERTICAL

        return ds, ds.shape

    @staticmethod
    def _concatenate(
        arrays: List[np.ndarray],
        axis: int,
        pad_ones: bool = False,
        enable_spu_cache: bool = False,
    ) -> np.ndarray:
        if pad_ones:
            if axis == 1:
                ones = jnp.ones((arrays[0].shape[0], 1), dtype=arrays[0].dtype)
            else:
                ones = jnp.ones((1, arrays[0].shape[1]), dtype=arrays[0].dtype)
            arrays.append(ones)
        x = jnp.concatenate(arrays, axis=axis)
        if enable_spu_cache:
            x = spu.experimental.make_cached_var(x)
        return x

    def _to_spu(self, d: Union[FedNdarray, VDataFrame]):
        if isinstance(d, VDataFrame):
            d = d.values
        return [d.partitions[pyu].to(self.spu) for pyu in d.partitions]
