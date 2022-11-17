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

from .core import pva


def pva_eval(
    actual: Union[FedNdarray, VDataFrame],
    prediction: Union[FedNdarray, VDataFrame],
    target,
) -> PYUObject:
    """Compute Prediction Vs Actual score.

    Args:
        actual: Union[FedNdarray, VDataFrame]

        prediction: Union[FedNdarray, VDataFrame]

        target: numeric
            the target label in actual entries to consider.
    compute:
        result: PYUObject
            Underlying a float of abs(mean(prediction) - sum(actual == target)/count(actual))
    """
    # for now we only consider vertical splitting case
    # y_true and y_score belongs to the same and single party
    assert isinstance(
        actual, (FedNdarray, VDataFrame)
    ), "actual should be FedNdarray or VDataFrame"
    assert isinstance(
        prediction, (FedNdarray, VDataFrame)
    ), "prediction should be FedNdarray or VDataFrame"
    assert (
        actual.shape == prediction.shape
    ), "actual and prediction should have the same shapes"
    assert (
        actual.shape[1] == 1
    ), "actual must be a single column, reshape before proceed"
    assert len(actual.partitions) == len(
        prediction.partitions
    ), "actual and prediction should have the same partitions"
    assert len(prediction.partitions) == 1, "y_score should have one partition"

    device1 = [*actual.partitions.keys()][0]
    device2 = [*prediction.partitions.keys()][0]
    assert (
        device1 == device2
    ), "Currently require the device for two inputs are the same"
    # Later may use spu

    device = device1
    if isinstance(actual, FedNdarray):
        actual = [*actual.partitions.values()][0]
    else:
        actual = ([*actual.partitions.values()][0]).data

    if isinstance(prediction, FedNdarray):
        prediction = [*prediction.partitions.values()][0]
    else:
        prediction = ([*prediction.partitions.values()][0]).data

    return device(pva)(actual, prediction, target)
