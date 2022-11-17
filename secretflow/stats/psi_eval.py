# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

# TODO: HDataFrame, VDataFrame and SPU support in future

from typing import Union

from secretflow.data import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYUObject

from .core import psi


def psi_eval(
    X: Union[FedNdarray, VDataFrame], Y: Union[FedNdarray, VDataFrame], split_points
) -> PYUObject:
    """Calculate population stability index.

    Args:
        X: Union[FedNdarray, VDataFrame]
            a collection of samples
        Y: Union[FedNdarray, VDataFrame]
            a collection of samples
        split_points: array
            an ordered sequence of split points
    Returns:
        result: float
            population stability index
    """
    assert isinstance(
        X, (FedNdarray, VDataFrame)
    ), "X should be FedNdarray or VDataFrame"
    assert isinstance(
        Y, (FedNdarray, VDataFrame)
    ), "Y should be FedNdarray or VDataFrame"
    assert X.shape[1] == 1, "X must be a single column, reshape before proceed"
    assert Y.shape[1] == 1, "Y must be a single column, reshape before proceed"
    assert len(X.partitions) == len(
        Y.partitions
    ), "X and Y should have the same partitions"
    assert len(Y.partitions) == 1, "Y should have one partition"

    device1 = [*X.partitions.keys()][0]
    device2 = [*Y.partitions.keys()][0]
    assert (
        device1 == device2
    ), "Currently require the device for two inputs are the same"
    # Later may use spu

    device = device1

    if isinstance(X, FedNdarray):
        X = [*X.partitions.values()][0]
    else:
        X = ([*X.partitions.values()][0]).data

    if isinstance(Y, FedNdarray):
        Y = [*Y.partitions.values()][0]
    else:
        Y = ([*Y.partitions.values()][0]).data
    return device(psi)(X, Y, split_points)
