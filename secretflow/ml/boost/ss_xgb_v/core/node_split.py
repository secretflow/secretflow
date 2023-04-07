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


from enum import Enum, unique
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from secretflow.utils import sigmoid as appr_sig


@unique
class RegType(Enum):
    Linear = 'linear'
    Logistic = 'logistic'


'''
stateless functions use in XGB model's node split.
please keep functions stateless to make jax happy
see https://jax.readthedocs.io/en/latest/jax-101/07-state.html
'''


def sigmoid(pred: np.ndarray) -> np.ndarray:
    return appr_sig.sr_sig(pred)


def compute_obj(
    G: np.ndarray,
    H: np.ndarray,
    reg_lambda: float,
) -> np.ndarray:
    '''
    compute objective values of input buckets.

    Args:
        G/H: sum of first and second order gradient in each bucket.
        reg_lambda: L2 regularization term

    Return:
        objective values.
    '''
    return (G / (H + reg_lambda)) * G


def compute_weight(
    G: float,
    H: float,
    reg_lambda: float,
    learning_rate: float,
) -> np.ndarray:
    '''
    compute weight values of tree leaf nodes.

    Args:
        G/H: sum of first and second order gradient in each node.
        reg_lambda: L2 regularization term
        learning_rate: Step size shrinkage used in update to prevents overfitting.

    Return:
        weight values.
    '''
    w = -((G / (H + reg_lambda)) * learning_rate)
    return jnp.select([H == 0], [0], w)


def compute_gh(
    y: np.ndarray, pred: np.ndarray, objective: RegType
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    compute first and second order gradient of each sample.

    Args:
        y: sample true label of each sample.
        pred: prediction of each sample.
        objective: regression learning objective,

    Return:
        weight values.
    '''
    if objective == RegType.Linear:
        g = pred - y
        h = jnp.ones(pred.shape)
    elif objective == RegType.Logistic:
        yhat = sigmoid(pred)
        g = yhat - y
        h = yhat * (1 - yhat)
    else:
        raise f"unknown objective {objective}"

    return g, h


def tree_setup(
    pred: np.ndarray,
    y: np.ndarray,
    sub_choices: np.ndarray,
    objective: RegType,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Set up pre-tree context.
    '''
    assert y.shape == pred.shape

    if sub_choices is not None:
        pred = pred[:, sub_choices]
        y = y[:, sub_choices]

    return compute_gh(y, pred, objective)


def compute_gradient_sums(
    nodes_s: List[np.ndarray],
    cache: List[List[np.ndarray]],
    col_choices: np.ndarray,
    sub_choices: np.ndarray,
    gh: List[np.ndarray],
    buckets_map: np.ndarray,
):
    # only compute the gradient sums of left children node.
    l_nodes_s = [s for idx, s in enumerate(nodes_s) if idx % 2 == 0]
    if cache:
        # and cache their parents' gradient sums
        # so we can get right children's sum by a simple subtraction
        GL_cache = cache[0]
        HL_cache = cache[1]
        assert len(GL_cache) == len(l_nodes_s)
    else:
        # root level, no cache
        GL_cache = None
        HL_cache = None

    if col_choices is not None:
        buckets_map = buckets_map[:, col_choices]
    if sub_choices is not None:
        buckets_map = buckets_map[sub_choices, :]

    # current level nodes' gradient sums of each buckets
    level_nodes_G = list()
    level_nodes_H = list()
    for idx, s in enumerate(l_nodes_s):
        if sub_choices is not None:
            s = s[:, sub_choices]
        g = gh[0] * s
        h = gh[1] * s
        # compute the gradient sums of each buckets in this node
        lchild_GL = jnp.matmul(g, buckets_map)
        lchild_HL = jnp.matmul(h, buckets_map)
        level_nodes_G.append(lchild_GL)
        level_nodes_H.append(lchild_HL)
        if GL_cache is not None:
            level_nodes_G.append(GL_cache[idx] - lchild_GL)
            level_nodes_H.append(HL_cache[idx] - lchild_HL)

    # gradient sums of left child nodes after splitting by each bucket
    GL = jnp.concatenate(level_nodes_G, axis=0)
    HL = jnp.concatenate(level_nodes_H, axis=0)

    return (GL, HL), (level_nodes_G, level_nodes_H)


def find_best_split_bucket(
    GHs: List[List[np.ndarray]],
    reg_lambda: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    '''
    compute the gradient sums of the containing instances in each split bucket
    and find best split bucket for each node which has the max split gain.

    Args:
        context: comparison context.
        nodes_s: sample select indexes of each node from same tree level.
        last_level: if this split is last level, next level is leaf nodes.

    Return:
        idx of split bucket for each node.
    '''
    GHs = list(zip(*GHs))

    # gradient sums of left child nodes after splitting by each bucket
    GL = sum(GHs[0])
    HL = sum(GHs[1])

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

    # gamma == 0
    gain = obj_l + obj_r - obj

    split_buckets = jnp.argmax(gain, axis=1)

    return split_buckets


def init_pred(base: float, samples: int):
    shape = (1, samples)
    return jnp.full(shape, base)


def root_select(samples: int) -> List[np.ndarray]:
    return (jnp.ones((1, samples), dtype=jnp.int8),)


def get_child_select(
    nodes_s: List[np.ndarray], lchilds_ss: List[np.ndarray], fragments: int
) -> List[np.ndarray]:
    '''
    compute the next level's select indexes.

    Args:
        nodes_s: sample select indexes of each node from current level's nodes.
        lchilds_ss: left children's sample select idx for current level's nodes.

    Return:
        sample select indexes for nodes in next tree level.
    '''
    lchilds_ss = list(zip(*lchilds_ss))
    lchilds_s = [jnp.concatenate(ss, axis=None) for ss in lchilds_ss]
    nodes_s = list(zip(*nodes_s))
    nodes_s = [jnp.concatenate(ns, axis=1) for ns in nodes_s]
    assert len(lchilds_s) == len(nodes_s), f"{len(lchilds_s)} != {len(nodes_s)}"
    childs_s = list()
    for current, lchile in zip(nodes_s, lchilds_s):
        assert current.size == lchile.size
        # current node's select mark with left child's select
        ls = current * lchile
        # get right child's select by sub.
        rs = current - ls
        childs_s.append(jnp.array_split(ls, fragments, axis=1))
        childs_s.append(jnp.array_split(rs, fragments, axis=1))

    return list(zip(*childs_s))


def predict_tree_weight(selects: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    '''
    get final pred for this tree.

    Args:
        selects: leaf nodes' sample selects from each model handler.
        weights: leaf weights in secure share.

    Return:
        pred
    '''
    select = selects[0]
    for i in range(1, len(selects)):
        select = select * selects[i]

    assert (
        select.shape[1] == weights.shape[0]
    ), f"select {select.shape}, weights {weights.shape}"

    return jnp.matmul(select, weights).reshape((1, select.shape[0]))


def get_weight(
    sums: List[List[np.ndarray]],
    reg_lambda: float,
    learning_rate: float,
) -> np.ndarray:
    sums = list(zip(*sums))

    g_sum = sum(sums[0])
    h_sum = sum(sums[1])

    return compute_weight(g_sum, h_sum, reg_lambda, learning_rate)


def sum_leaf(ss: List[np.ndarray], gh: List[np.ndarray], sub_choices: np.ndarray):
    s = jnp.concatenate(ss, axis=0)

    if sub_choices is not None:
        s = s[:, sub_choices]

    g_sum = jnp.matmul(s, jnp.transpose(gh[0]))
    h_sum = jnp.matmul(s, jnp.transpose(gh[1]))

    return (g_sum, h_sum)


def update_train_pred(pred: List[np.ndarray], current: np.ndarray, fragments: int):
    assert len(pred) == fragments
    current = jnp.array_split(current, fragments, axis=1)
    for idx in range(fragments):
        assert pred[idx].shape == current[idx].shape
        pred[idx] = pred[idx] + current[idx]

    return pred
