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

from typing import List, Tuple

import numpy as np


def compute_obj(G: np.ndarray, H: np.ndarray, reg_lambda: float) -> np.ndarray:
    """
    compute objective values of input buckets.

    Args:
        G/H: sum of first and second order gradient in each bucket.
        reg_lambda: L2 regularization term

    Return:
        objective values.
    """
    return (G / (H + reg_lambda)) * G


def compute_weight_from_node_select(
    node_select: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    reg_lambda: float,
    learning_rate: float,
) -> np.ndarray:
    g_sum = np.matmul(node_select, g)
    h_sum = np.matmul(node_select, h)

    return compute_weight(g_sum, h_sum, reg_lambda, learning_rate)


def compute_weight(
    G: float, H: float, reg_lambda: float, learning_rate: float
) -> np.ndarray:
    """
    compute weight values of tree leaf nodes.

    Args:
        G/H: sum of first and second order gradient in each node.
        reg_lambda: L2 regularization term
        learning_rate: Step size shrinkage used in update to prevents overfitting.

    Return:
        weight values.
    """
    w = -((G / (H + reg_lambda)) * learning_rate)
    return np.select([H == 0], [0], w)


def calculate_gains(
    level_nodes_G: List[np.ndarray], level_nodes_H: List[np.ndarray], reg_lambda: float
) -> np.ndarray:
    GL = np.concatenate(level_nodes_G, axis=0)
    HL = np.concatenate(level_nodes_H, axis=0)

    # last buckets is the total gradient sum of all samples belong to current level nodes.
    GA = GL[:, -1].reshape(-1, 1)
    HA = HL[:, -1].reshape(-1, 1)
    # gradient sums of right child nodes after splitting by each bucket
    GR = GA - GL
    HR = HA - HL
    obj_l = compute_obj(GL, HL, reg_lambda)
    obj_r = compute_obj(GR, HR, reg_lambda)

    # last objective value means split all sample to left, equal to no split.
    obj = obj_l[:, -1].reshape(-1, 1)
    gain = obj_l + obj_r - obj
    return gain


def find_best_splits(
    level_nodes_G: List[np.array],
    level_nodes_H: List[np.array],
    reg_lambda: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """find the best split buckets and if gains > gamma"""
    gain = calculate_gains(level_nodes_G, level_nodes_H, reg_lambda)
    split_buckets = np.argmax(gain, 1)
    should_split = (
        (np.max(gain, 1) - gamma) > 0
        if gamma > 0
        else np.ones(split_buckets.shape).astype(bool)
    )
    return (split_buckets, should_split)


def find_best_splits_with_gains(
    nodes_G: List[np.array], nodes_H: List[np.array], reg_lambda: float, gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gain = calculate_gains(nodes_G, nodes_H, reg_lambda)
    split_buckets = np.argmax(gain, 1)
    split_gains = np.max(gain, 1)
    should_split = (split_gains - gamma) > 0
    return (split_buckets, split_gains, should_split)


def find_single_best_split(
    node_indices: List[int],
    split_buckets: np.ndarray,
    split_gains: np.ndarray,
) -> Tuple[int, int, float]:
    index = np.argmax(split_gains)
    return (node_indices[index], split_buckets[index], split_gains[index])
