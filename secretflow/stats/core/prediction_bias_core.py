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

# This is a single party based prediction vs actual calculation

import logging
import math
from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Union

import jax.numpy as jnp
import pandas as pd


@unique
class PredictionBiasBucketMethod(Enum):
    EQUAL_FREQUENCY = 'equal_frequency'
    EQUAL_WIDTH = 'equal_width'


@dataclass
class BucketPredictionBiasReport:
    left_endpoint: float
    left_closed: bool
    right_endpoint: float
    right_closed: bool
    isna: bool
    avg_prediction: float = None
    avg_label: float = None
    bias: float = None
    absolute: bool = None


@dataclass
class PredictionBiasReport:
    buckets: List[BucketPredictionBiasReport]


def prediction_bias(
    prediction: Union[pd.DataFrame, jnp.array],
    label: Union[pd.DataFrame, jnp.array],
    bucket_num: int = 1,
    absolute: bool = True,
    bucket_method: PredictionBiasBucketMethod = PredictionBiasBucketMethod.EQUAL_WIDTH,
    min_item_cnt_per_bucket: int = None,
):
    """prediction bias = average of predictions - average of labels.

    Args:
        prediction (Union[pd.DataFrame, jnp.array]): prediction input.
        label (Union[pd.DataFrame, jnp.array]): label input
        bucket_num (int, optional): num of bucket. Defaults to 1.
        absolute (bool, optional): whether to output absolute value of bias. Defaults to True.
        bucket_method (PredictionBiasBucketMethod, optional): bucket method. Defaults to PredictionBiasBucketMethod.EQUAL_WIDTH.
        min_item_cnt_per_bucket (int): min item cnt per bucket. If any bucket doesn't meet the requirement, error raises.

    Returns:
        _type_: _description_
    """
    if isinstance(label, pd.DataFrame):
        label = label.to_numpy()
    if isinstance(prediction, pd.DataFrame):
        prediction = prediction.to_numpy()
    assert label.size > 1, "there must be at least one actual"
    assert (
        prediction.size == label.size
    ), "there must be at equal number of actuals and predictions"

    if bucket_num < 1:
        bucket_num = 1
        logging.warning('bucket_num is less than 1. Changed to 1.')

    if bucket_num > prediction.size:
        bucket_num = prediction.size
        logging.warning(
            'bucket_num is greater than number of prediction. Changed to number of prediction.'
        )

    index = jnp.argsort(prediction, axis=0)
    prediction = jnp.take_along_axis(prediction, index, axis=0)
    label = jnp.take_along_axis(label, index, axis=0)
    report = PredictionBiasReport(buckets=[])

    if bucket_method == PredictionBiasBucketMethod.EQUAL_WIDTH:
        hist, bin_edgesarray = jnp.histogram(prediction, bucket_num)

        total_cnt = 0
        for i, cnt in enumerate(hist):
            cnt = int(cnt)
            left_endpoint = bin_edgesarray[i]
            left_closed = True
            right_endpoint = bin_edgesarray[i + 1]
            right_closed = i == hist.size - 1

            if min_item_cnt_per_bucket is not None:
                if cnt < min_item_cnt_per_bucket and cnt > 0:
                    raise RuntimeError(
                        f"One bin doesn't meet min_item_cnt_per_bucket requirement."
                    )

            if cnt == 0:
                report.buckets.append(
                    BucketPredictionBiasReport(
                        left_endpoint=float(left_endpoint),
                        left_closed=left_closed,
                        right_endpoint=float(right_endpoint),
                        right_closed=right_closed,
                        isna=True,
                        avg_prediction=0,
                        avg_label=0,
                        bias=0,
                        absolute=absolute,
                    )
                )
            else:
                avg_prediction = jnp.average(prediction[total_cnt : total_cnt + cnt])
                avg_label = jnp.average(label[total_cnt : total_cnt + cnt])

                bias = avg_prediction - avg_label
                if absolute:
                    bias = jnp.abs(bias)
                report.buckets.append(
                    BucketPredictionBiasReport(
                        left_endpoint=float(left_endpoint),
                        left_closed=left_closed,
                        right_endpoint=float(right_endpoint),
                        right_closed=right_closed,
                        isna=False,
                        avg_prediction=float(avg_prediction),
                        avg_label=float(avg_label),
                        bias=float(bias),
                        absolute=absolute,
                    )
                )

            total_cnt += cnt
    else:
        bucket_size = math.ceil(prediction.size / bucket_num)

        if min_item_cnt_per_bucket is not None:
            if (
                bucket_size < min_item_cnt_per_bucket
                or (prediction.size - bucket_size * (bucket_num - 1))
                < min_item_cnt_per_bucket
            ):
                raise RuntimeError(
                    f"One bin doesn't meet min_item_cnt_per_bucket requirement."
                )

        for i in range(bucket_num):
            total_cnt = bucket_size * i
            left_endpoint = prediction[total_cnt]
            left_closed = True

            if i == bucket_num - 1:
                right_endpoint = prediction[prediction.size - 1]
                right_closed = True

                avg_prediction = jnp.average(prediction[total_cnt : prediction.size])
                avg_label = jnp.average(label[total_cnt : prediction.size])

            else:
                right_endpoint = prediction[total_cnt + bucket_size]
                right_closed = False

                avg_prediction = jnp.average(
                    prediction[total_cnt : total_cnt + bucket_size]
                )
                avg_label = jnp.average(label[total_cnt : total_cnt + bucket_size])

            bias = avg_prediction - avg_label

            if absolute:
                bias = jnp.abs(bias)

            report.buckets.append(
                BucketPredictionBiasReport(
                    left_endpoint=float(left_endpoint),
                    left_closed=left_closed,
                    right_endpoint=float(right_endpoint),
                    right_closed=i == bucket_num - 1,
                    isna=False,
                    avg_prediction=float(avg_prediction),
                    avg_label=float(avg_label),
                    bias=float(bias),
                    absolute=absolute,
                )
            )

    return report
