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


from typing import Any, Dict, List, Tuple
import jax.numpy as jnp
from enum import Enum, unique
import numpy as np
import math
import jax

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


def compute_obj(G: np.ndarray, H: np.ndarray, reg_lambda: float) -> np.ndarray:
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
    G: float, H: float, reg_lambda: float, learning_rate: float
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


def get_weight(context: Dict[str, Any], s: np.ndarray) -> np.ndarray:
    '''
    compute weight values of tree leaf nodes.

    Args:
        context: comparison context.
        s: sample selects in each leaf node.

    Return:
        weight values.
    '''
    if 'sub_choices' in context:
        s = s[:, context['sub_choices']]

    g_sum = jnp.matmul(s, jnp.transpose(context['g']))
    h_sum = jnp.matmul(s, jnp.transpose(context['h']))

    return compute_weight(g_sum, h_sum, context['reg_lambda'], context['learning_rate'])


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


def global_setup(
    buckets_map: List[np.ndarray],
    y: np.ndarray,
    seed: int,
    reg_lambda: float,
    learning_rate: float,
) -> Dict[str, Any]:
    '''
    Set up global context.
    '''
    context = dict()
    context['buckets_map'] = jnp.concatenate(buckets_map, axis=1)
    # transpose 2D-array or reshape 1D-array to 2D
    context['y'] = y.reshape((1, y.shape[0]))
    context['prng_key'] = jax.random.PRNGKey(seed)
    context['reg_lambda'] = reg_lambda
    context['learning_rate'] = learning_rate

    return context


def tree_setup(
    context: Dict[str, Any],
    pred: np.ndarray,
    col_choices: List[np.ndarray],
    objective: RegType,
    samples: int,
    subsample: float,
) -> Dict[str, Any]:
    '''
    Set up pre-tree context.
    '''
    if len(col_choices):
        context['col_choices'] = jnp.concatenate(col_choices, axis=None)

    y = context['y']

    choices = math.ceil(samples * subsample)
    assert choices > 0, f"subsample {subsample} is too small"
    if choices < samples:
        key = context['prng_key']
        sub_choices = jax.random.permutation(key, samples, independent=True)[:choices]
        context['sub_choices'] = sub_choices
        # update key for next round
        context['prng_key'], _ = jax.random.split(key)

        pred = pred[:, sub_choices]
        y = y[:, sub_choices]

    gh = compute_gh(y, pred, objective)
    context['g'] = gh[0]
    context['h'] = gh[1]

    return context


def find_best_split_bucket(
    context: Dict[str, Any], nodes_s: List[np.ndarray], last_level: bool
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
    # only compute the gradient sums of left children node.
    l_nodes_s = [s for idx, s in enumerate(nodes_s) if idx % 2 == 0]
    if 'cache' in context:
        # and cache their parents' gradient sums
        # so we can get right children's sum by a simple subtraction
        GL_cache = context['cache'][0]
        HL_cache = context['cache'][1]
        assert len(GL_cache) == len(l_nodes_s)
    else:
        # root level, no cache
        GL_cache = None
        HL_cache = None

    buckets_map = context['buckets_map']
    if 'col_choices' in context:
        buckets_map = buckets_map[:, context['col_choices']]
    if 'sub_choices' in context:
        buckets_map = buckets_map[context['sub_choices'], :]

    # current level nodes' gradient sums of each buckets
    level_nodes_G = list()
    level_nodes_H = list()
    for idx, s in enumerate(l_nodes_s):
        if 'sub_choices' in context:
            s = s[:, context['sub_choices']]
        sg = context['g'] * s
        sh = context['h'] * s
        # compute the gradient sums of each buckets in this node
        lchild_GL = jnp.matmul(sg, buckets_map)
        lchild_HL = jnp.matmul(sh, buckets_map)
        level_nodes_G.append(lchild_GL)
        level_nodes_H.append(lchild_HL)
        if GL_cache is not None:
            level_nodes_G.append(GL_cache[idx] - lchild_GL)
            level_nodes_H.append(HL_cache[idx] - lchild_HL)

    if not last_level:
        context['cache'] = (level_nodes_G, level_nodes_H)
    elif 'cache' in context:
        del context['cache']

    # gradient sums of left child nodes after splitting by each bucket
    GL = jnp.concatenate(level_nodes_G, axis=0)
    HL = jnp.concatenate(level_nodes_H, axis=0)

    # last buckets is the total gradient sum of all samples belong to current level nodes.
    GA = GL[:, -1].reshape(-1, 1)
    HA = HL[:, -1].reshape(-1, 1)

    # gradient sums of right child nodes after splitting by each bucket
    GR = GA - GL
    HR = HA - HL

    obj_l = compute_obj(GL, HL, context['reg_lambda'])
    obj_r = compute_obj(GR, HR, context['reg_lambda'])

    # last objective value means split all sample to left, equal to no split.
    obj = obj_l[:, -1].reshape(-1, 1)

    # gamma == 0
    gain = obj_l + obj_r - obj

    split_buckets = jnp.argmax(gain, axis=1)

    return split_buckets, context


def init_pred(base: float, samples: int):
    shape = (1, samples)
    return jnp.full(shape, base)


def root_select(samples: int) -> List[np.ndarray]:
    root_select = jnp.ones((1, samples), dtype=jnp.int8)
    return (root_select,)


def get_child_select(
    nodes_s: List[np.ndarray], lchilds_ss: List[np.ndarray]
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
    assert len(lchilds_s) == len(nodes_s), f"{len(lchilds_s)} != {len(nodes_s)}"
    childs_s = list()
    for current, lchile in zip(nodes_s, lchilds_s):
        assert current.size == lchile.size
        # current node's select mark with left child's select
        ls = current * lchile
        # get right child's select by sub.
        rs = current - ls
        childs_s.append(ls)
        childs_s.append(rs)
    return childs_s


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


def do_leaf(context: Dict[str, Any], ss: List[np.ndarray]) -> np.ndarray:
    # see get_weight.
    s = jnp.concatenate(ss, axis=0)
    return get_weight(context, s)
