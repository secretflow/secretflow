# Copyright 2023 Ant Group Co., Ltd.
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

import math
from typing import Tuple, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype

from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYUObject, reveal, wait


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
        "for horizontal dataset please use secretflow_fl.ml.boost.homo_boost"  # TODO by jzc: modify
    )

    shape = ds.shape
    assert math.prod(shape), f"not support empty dataset, shape {shape}"
    return ds, shape


def non_empty(x, worker):
    return worker(lambda x: x.size > 0)(x)


def validate(
    dataset, label
) -> Tuple[FedNdarray, Tuple[int, int], PYUObject, Tuple[int, int]]:
    x, x_shape = prepare_dataset(dataset)
    y, y_shape = prepare_dataset(label)
    assert len(x_shape) == 2, "only support 2D-array on dtrain"

    data_check_task = [
        worker(data_checks)(x_val, worker) for worker, x_val in x.partitions.items()
    ]

    data_not_emmpty = reveal(
        [non_empty(x_val, worker) for worker, x_val in x.partitions.items()]
    )
    to_remove_devices = []
    for i, empty_device in enumerate(x.partitions.keys()):
        if not data_not_emmpty[i]:
            to_remove_devices.append(empty_device)

    for device in to_remove_devices:
        x.partitions.pop(device)

    assert len(y_shape) == 1 or y_shape[1] == 1, "label only support one label col"
    samples = y_shape[0]
    assert samples == x_shape[0], "dtrain & label are not aligned"
    assert len(y.partitions) == 1, "label only support one partition"
    # get y as a PYUObject
    y = list(y.partitions.values())[0]
    y = y.device(lambda y: y.reshape(-1, 1, order='F'))(y)
    y_shape = (samples, 1)
    assert samples > 0, "cannot have empty samples"
    wait(data_check_task)
    return x, x_shape, y, y_shape


def validate_tweedie_label(y: PYUObject):
    """Tweedie regression, y label must be non-negative"""
    wait(y.device(check_not_negative)(y, y.device.party))


def validate_sample_weight(
    sample_weight: Union[FedNdarray, VDataFrame], y_shape: Tuple
) -> Union[None, PYUObject]:
    if sample_weight is None:
        return None
    sample_weight, shape = prepare_dataset(sample_weight)
    assert len(shape) == 1 or shape[1] == 1, "label only support one label col"
    assert (
        shape[0] == y_shape[0]
    ), f"sample weight should have shape like y {y_shape}, got {shape}"
    sample_weight = list(sample_weight.partitions.values())[0]
    sample_weight = sample_weight.device(
        lambda w: w.reshape(
            -1,
        )
    )(sample_weight)
    return sample_weight


def data_checks(x, worker):
    check_numeric(x, worker)
    check_null_val(x, worker)


def check_not_negative(x, worker):
    assert (
        x >= 0
    ).all(), "worker {}'s data contains negative values, which is not allowed.".format(
        worker
    )


def check_null_val(x, worker):
    assert ~pd.isnull(
        x
    ).any(), "worker {}'s data contain NaN or None, this may cause errors or degraded performance in stages like qcut.".format(
        worker
    )


def check_numeric(x, worker):
    assert is_numeric_dtype(
        x
    ), "worker {}'s data is not numeric type, encode feature before proceed.".format(
        worker
    )
