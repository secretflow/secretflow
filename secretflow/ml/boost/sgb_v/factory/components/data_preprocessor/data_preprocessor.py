# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

from typing import Tuple, Union
from secretflow.device import PYUObject
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame

from ..component import Component


def prepare_dataset(
    ds: Union[FedNdarray, VDataFrame]
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
    ), f"ds should be FedNdarray or VDataFrame, got {type(ds)}"

    ds = ds if isinstance(ds, FedNdarray) else ds.values

    assert ds.partition_way == PartitionWay.VERTICAL, (
        "Only support vertical dataset, "
        "for horizontal dataset please use secreflow.ml.boost.homo_boost"
    )

    shape = ds.shape
    assert math.prod(shape), f"not support empty dataset, shape {shape}"

    return ds, shape


class DataPreprocessor(Component):
    def __init__(self) -> None:
        super().__init__()

    def show_params(self):
        return

    def set_params(self, _):
        return

    def get_params(self, _):
        return

    def set_devices(self, _):
        return

    def validate(
        self, dataset, label
    ) -> Tuple[FedNdarray, Tuple[int, int], PYUObject, Tuple[int, int]]:
        x, x_shape = prepare_dataset(dataset)
        y, y_shape = prepare_dataset(label)
        assert len(x_shape) == 2, "only support 2D-array on dtrain"
        assert len(y_shape) == 1 or y_shape[1] == 1, "label only support one label col"
        samples = y_shape[0]
        assert samples == x_shape[0], "dtrain & label are not aligned"
        assert len(y.partitions) == 1, "label only support one partition"
        # get y as a PYUObject
        y = list(y.partitions.values())[0]
        return x, x_shape, y, y_shape
