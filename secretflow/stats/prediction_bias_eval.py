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
from secretflow.stats.core.prediction_bias_core import (
    PredictionBiasBucketMethod,
    prediction_bias,
)


def prediction_bias_eval(
    prediction: Union[FedNdarray, VDataFrame],
    label: Union[FedNdarray, VDataFrame],
    bucket_num: int,
    absolute: bool,
    bucket_method: str,
    min_item_cnt_per_bucket: int = None,
) -> PYUObject:
    """prediction bias = average of predictions - average of labels.

    Args:
        prediction (Union[FedNdarray, VDataFrame]): prediction input.
        label (Union[FedNdarray, VDataFrame]): abel input
        bucket_num (int): num of bucket.
        absolute (bool): whether to output absolute value of bias.
        bucket_method (str): bucket method.
        min_item_cnt_per_bucket(int): min item cnt per bucket. If any bucket doesn't meet the requirement, error raises.

    Returns:
        PYUObject: PredictionBiasReport
    """
    assert isinstance(
        label, (FedNdarray, VDataFrame)
    ), "label should be FedNdarray or VDataFrame"
    assert isinstance(
        prediction, (FedNdarray, VDataFrame)
    ), "prediction should be FedNdarray or VDataFrame"
    assert bucket_method in [
        'equal_frequency',
        'equal_width',
    ], "bucket_method must be in ['equal_frequency', 'equal_width']"
    assert (
        label.shape == prediction.shape
    ), f"label {label.shape} and prediction {prediction.shape} should have the same shape"
    assert label.shape[1] == 1, "label must be a single column, reshape before proceed"
    assert len(label.partitions) == len(
        prediction.partitions
    ), "label and prediction should have the same partitions"
    assert len(prediction.partitions) == 1, "y_score should have one partition"

    device1 = [*label.partitions.keys()][0]
    device2 = [*prediction.partitions.keys()][0]
    assert (
        device1 == device2
    ), "Currently we requires both inputs belongs to the same party and computation happens locally."

    # Later may use spu

    device = device1
    if isinstance(label, FedNdarray):
        label = [*label.partitions.values()][0]
    else:
        label = ([*label.partitions.values()][0]).data

    if isinstance(prediction, FedNdarray):
        prediction = [*prediction.partitions.values()][0]
    else:
        prediction = ([*prediction.partitions.values()][0]).data

    bucket_method = (
        PredictionBiasBucketMethod.EQUAL_FREQUENCY
        if bucket_method == 'equal_frequency'
        else PredictionBiasBucketMethod.EQUAL_WIDTH
    )

    return device(prediction_bias)(
        prediction, label, bucket_num, absolute, bucket_method, min_item_cnt_per_bucket
    )
