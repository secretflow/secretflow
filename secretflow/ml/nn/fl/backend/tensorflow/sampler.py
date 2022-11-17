#!/usr/bin/env python3
# *_* coding: utf-8 *_*

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


import logging
import math

import numpy as np
import tensorflow as tf

from secretflow.data.horizontal.sampler import PoissonDataSampler


def batch_sampler(
    x, y, s_w, sampling_rate, buffer_size, shuffle, repeat_count, random_seed
):
    """
    implementation of batch sampler

    Args:
        x: feature, FedNdArray or HDataFrame
        y: label, FedNdArray or HDataFrame
        s_w: sample weight of this dataset
        sampling_rate: Sampling rate of a batch
        buffer_size: shuffle size
        shuffle: A bool that indicates whether the input should be shuffled
        repeat_count: num of repeats
        random_seed: Prg seed for shuffling
    Returns:
        data_set: tf.data.Dataset
    """
    batch_size = math.floor(x.shape[0] * sampling_rate)
    assert batch_size > 0, "Unvalid batch size"

    x = [x]
    if y is not None and len(y.shape) > 0:
        x.append(y.astype(np.float64))
    if s_w is not None and len(s_w.shape) > 0:
        x.append(s_w.astype(np.float64))
    x = tuple(x)

    data_set = (
        tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(repeat_count)
    )
    if shuffle:
        if buffer_size is None:
            buffer_size = batch_size * 8
        data_set = data_set.shuffle(buffer_size, seed=random_seed)
    return data_set


def possion_sampler(x, y, s_w, sampling_rate, random_seed):
    """
    implementation of possion sampler

    Args:
        x: feature, FedNdArray or HDataFrame
        y: label, FedNdArray or HDataFrame
        s_w: sample weight of this dataset
        sampling_rate: Sampling rate of a batch
        random_seed: Prg seed for shuffling
    Returns:
        data_set: tf.data.Dataset
    """
    gen = PoissonDataSampler(x, y, s_w, sampling_rate)
    gen.set_random_seed(random_seed)
    x_shape = list(x.shape)
    x_shape[0] = None
    y_shape = list(y.shape)
    y_shape[0] = None
    if s_w is not None:
        s_w_shape = list(s_w.shape)
        s_w_shape[0] = None
        data_set = tf.data.Dataset.from_generator(
            lambda: gen,
            output_signature=(
                tf.TensorSpec(shape=x_shape, dtype=x.dtype),
                tf.TensorSpec(shape=y_shape, dtype=y.dtype),
                tf.TensorSpec(shape=s_w_shape, dtype=s_w.dtype),
            ),
        )
    else:
        data_set = tf.data.Dataset.from_generator(
            lambda: gen,
            output_signature=(
                tf.TensorSpec(shape=x_shape, dtype=x.dtype),
                tf.TensorSpec(shape=y_shape, dtype=y.dtype),
            ),
        )
    data_set = data_set.repeat().prefetch(tf.data.experimental.AUTOTUNE)
    return data_set


def sampler_data(
    sampler_method="batch",
    x=None,
    y=None,
    s_w=None,
    sampling_rate=None,
    buffer_size=None,
    shuffle=False,
    repeat_count=1,
    random_seed=1234,
):
    """
    do sample data by sampler_method

    Args:
        x: feature, FedNdArray or HDataFrame
        y: label, FedNdArray or HDataFrame
        s_w: sample weight of this dataset
        sampling_rate: Sampling rate of a batch
        buffer_size: shuffle size
        shuffle: A bool that indicates whether the input should be shuffled
        repeat_count: num of repeats
        random_seed: Prg seed for shuffling
    Returns:
        data_set: tf.data.Dataset
    """
    if sampler_method == "batch":
        data_set = batch_sampler(
            x, y, s_w, sampling_rate, buffer_size, shuffle, repeat_count, random_seed
        )
    elif sampler_method == "possion":
        data_set = possion_sampler(x, y, s_w, sampling_rate, random_seed)
    else:
        logging.error(f'Unvalid sampler {sampler_method} during building local dataset')
    return data_set
