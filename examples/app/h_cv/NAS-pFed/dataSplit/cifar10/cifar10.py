# Copyright 2024 Ant Group Co., Ltd.
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

import os.path
from datetime import datetime

import numpy as np
from torchvision import datasets

import random
import torch

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NUM_CLASSES = 10
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
TRAIN_EXAMPLES_PER_LABEL = int(TRAIN_EXAMPLES / 10)
TEST_EXAMPLES_PER_LABEL = int(TEST_EXAMPLES / 10)

current_dir = os.path.dirname(os.path.abspath(__file__))


def split_cifar10(
    dirichlet_parameter: float = 0.1,
    total_clients: int = 500,
):
    train_dataset = datasets.CIFAR10(root="./dataset", train=True, download=False)
    test_dataset = datasets.CIFAR10(root="./dataset", train=False, download=False)

    train_indices_by_class = [[] for _ in range(10)]
    test_indices_by_class = [[] for _ in range(10)]

    for idx, (_, label) in enumerate(train_dataset):
        train_indices_by_class[label].append(idx)
    for idx, (_, label) in enumerate(test_dataset):
        test_indices_by_class[label].append(idx)
    train_indices_by_class = np.array(train_indices_by_class)
    test_indices_by_class = np.array(test_indices_by_class)

    train_multinomial_vals = []
    test_multinomial_vals = []
    for i in range(total_clients):
        proportion = np.random.dirichlet(
            dirichlet_parameter
            * np.ones(
                NUM_CLASSES,
            )
        )
        train_multinomial_vals.append(proportion)
        test_multinomial_vals.append(proportion)
    train_multinomial_vals = np.array(train_multinomial_vals)
    test_multinomial_vals = np.array(test_multinomial_vals)

    train_client_samples = [[] for _ in range(total_clients)]
    test_client_samples = [[] for _ in range(total_clients)]
    train_count = np.zeros(NUM_CLASSES).astype(int)
    test_count = np.zeros(NUM_CLASSES).astype(int)

    train_examples_per_client = int(TRAIN_EXAMPLES / total_clients)
    test_examples_per_client = int(TEST_EXAMPLES / total_clients)

    for k in range(total_clients):
        for i in range(train_examples_per_client):
            sampled_label = np.argwhere(
                np.random.multinomial(1, train_multinomial_vals[k, :]) == 1
            )[0][0]
            train_client_samples[k].append(
                train_indices_by_class[sampled_label, train_count[sampled_label]]
            )
            train_count[sampled_label] += 1
            if train_count[sampled_label] == TRAIN_EXAMPLES_PER_LABEL:
                train_multinomial_vals[:, sampled_label] = 0
                train_multinomial_vals = (
                    train_multinomial_vals / train_multinomial_vals.sum(axis=1)[:, None]
                )
        for i in range(test_examples_per_client):
            sampled_label = np.argwhere(
                np.random.multinomial(1, test_multinomial_vals[k, :]) == 1
            )[0][0]
            test_client_samples[k].append(
                test_indices_by_class[sampled_label, test_count[sampled_label]]
            )
            test_count[sampled_label] += 1
            if test_count[sampled_label] == TEST_EXAMPLES_PER_LABEL:
                test_multinomial_vals[:, sampled_label] = 0
                test_multinomial_vals = (
                    test_multinomial_vals / test_multinomial_vals.sum(axis=1)[:, None]
                )
    return train_client_samples, test_client_samples
