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
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


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
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)  # set random seed for cpu
        torch.cuda.manual_seed(random_seed)  # set random seed for cuda
        torch.backends.cudnn.deterministic = True
    data_list = [torch.Tensor((x.astype(np.float64)).copy())]
    if y is not None and len(y.shape) > 0:
        data_list.append(torch.Tensor(y.copy()))
    if s_w is not None and len(s_w.shape) > 0:
        data_list.append(torch.Tensor(s_w.copy()))
    dataset = TensorDataset(*data_list)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )  # create dataloader

    return dataloader


# TODO: Need to be implemented
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
        dataloader: tf.data.Dataset
    """
    raise Exception("Possion sampler is not supported yet")


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
    else:
        logging.error(f'Unvalid sampler {sampler_method} during building local dataset')
    return data_set
