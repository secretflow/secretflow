# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
import logging
from typing import Dict, Union, Tuple

import jax.numpy as jnp

from .core.node_split import RegType
from .core.utils import prepare_dataset
from .core import node_split as split_fn
from .core.tree_worker import XgbTreeWorker as Worker

from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device.device.base import MoveConfig
from secretflow.device import (
    SPU,
    PYU,
    PYUObject,
    SPUObject,
    wait,
    SPUCompilerNumReturnsPolicy,
)


class XgbModel:
    '''
    SS Xgb Model & predict.
    '''

    def __init__(self, spu: SPU, objective: RegType, base: float) -> None:
        self.spu = spu
        self.objective = objective
        self.base = base
        # List[Dict[PYU, PYUObject of XgbTree]], owned by pyu, only knows split value if feature belong to this pyu.
        self.trees = list()
        # List[SPUObject of np.array], owned by spu and not reveal to any one
        self.weights = list()
        # TODO how to ser/der ?

    def _tree_pred(self, tree: Dict[PYU, PYUObject], weight: SPUObject) -> SPUObject:
        assert len(tree) == len(self.x)

        weight_selects = list()
        for worker in self.workers:
            device = worker.device
            assert device in tree
            s = worker.predict_weight_select(self.x[device].data, tree[device])
            weight_selects.append(s.to(self.spu))

        pred = self.spu(split_fn.predict_tree_weight)(weight_selects, weight)
        return pred

    def predict(
        self,
        dtrain: Union[FedNdarray, VDataFrame],
        to_pyu: PYU = None,
    ) -> Union[SPUObject, FedNdarray]:
        '''
        predict on dtrain with this model.

        Args:

            dtrain : [FedNdarray, VDataFrame]
                vertical split dataset.

            to: the prediction initiator
                if not None predict result is reveal to to_pyu device and save as FedNdarray
                otherwise, keep predict result in secret and save as SPUObject.

        Return:
            Pred values store in spu object or FedNdarray.
        '''
        if len(self.trees) == 0:
            return None
        x, _ = prepare_dataset(dtrain)
        assert len(x.partitions) == len(self.trees[0])
        self.workers = [Worker(0, device=pyu) for pyu in x.partitions]
        self.x = x.partitions
        preds = []
        for idx in range(len(self.trees)):
            pred = self._tree_pred(self.trees[idx], self.weights[idx])
            wait([pred])
            preds.append(pred)

        pred = self.spu(
            lambda ps, base: (jnp.sum(jnp.concatenate(ps, axis=0), axis=0) + base).reshape(-1, 1)
        )(preds, self.base)

        if self.objective == RegType.Logistic:
            pred = self.spu(split_fn.sigmoid)(pred)

        if to_pyu is not None:
            assert isinstance(to_pyu, PYU)
            return FedNdarray(
                partitions={
                    to_pyu: pred.to(to_pyu),
                },
                partition_way=PartitionWay.VERTICAL,
            )
        else:
            return pred


class Xgb:
    '''
    This method provides both classification and regression tree boosting (also known as GBDT, GBM)
    for vertical split dataset setting by using secret sharing.

    SS-XGB is short for secret sharing XGB.
    more details: https://arxiv.org/pdf/2005.08479.pdf

    Args:
        spu: secret device running MPC protocols

    '''

    def __init__(self, spu: SPU) -> None:
        # todo: distributed XGB, work with multiple spu to support large dataset.
        self.spu = spu

    def _update_pred(self, tree: Dict[PYU, PYUObject], weight: SPUObject) -> None:
        assert len(tree) == len(self.x)

        weight_selects = list()
        for worker in self.workers:
            device = worker.device
            assert device in tree
            s = worker.predict_weight_select(self.x[device].data, tree[device])
            weight_selects.append(s.to(self.spu))

        current = self.spu(split_fn.predict_tree_weight)(weight_selects, weight)
        self.pred = self.spu(lambda x, y: x + y)(self.pred, current)

    def _prepare(
        self,
        params: Dict,
        dataset: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
    ) -> None:
        x, x_shape = prepare_dataset(dataset)
        y, y_shape = prepare_dataset(label)
        assert len(x_shape) == 2, "only support 2D-array on dtrain"
        assert len(y_shape) == 1 or y_shape[1] == 1, "label only support one col"
        self.samples = y_shape[0]
        assert self.samples == x_shape[0], "dtrain & label are not aligned"
        assert len(y.partitions) == 1, "label only support one partition"

        self.y = list(y.partitions.values())[0]
        self.workers = [Worker(idx, device=pyu) for idx, pyu in enumerate(x.partitions)]
        self.x = x.partitions

        self.trees = int(params.pop('num_boost_round', 10))
        assert (
            1 <= self.trees <= 1024
        ), f"num_boost_round should in [1, 1024], got {self.trees}"

        self.depth = int(params.pop('max_depth', 5))
        assert (
            self.depth > 0 and self.depth <= 16
        ), f"max_depth should in [1, 16], got {self.depth}"

        self.lr = float(params.pop('learning_rate', 0.3))
        assert (
            self.lr > 0 and self.lr <= 1
        ), f"learning_rate should in (0, 1], got {self.lr}"

        obj = params.pop('objective', 'logistic')
        assert obj in [
            e.value for e in RegType
        ], f"objective should in {[e.value for e in RegType]}, got {obj}"
        self.obj = RegType(obj)

        self.reg_lambda = float(params.pop('reg_lambda', 0.1))
        assert (
            self.reg_lambda >= 0 and self.reg_lambda <= 10000
        ), f"reg_lambda should in [0, 10000], got {self.reg_lambda}"

        self.subsample = float(params.pop('subsample', 1))
        assert (
            self.subsample > 0 and self.subsample <= 1
        ), f"subsample should in (0, 1], got {self.subsample}"

        self.colsample = float(params.pop('colsample_bytree', 1))
        assert (
            self.colsample > 0 and self.colsample <= 1
        ), f"colsample_bytree should in (0, 1], got {self.colsample}"

        self.base = float(params.pop('base_score', 0))

        sketch = params.pop('sketch_eps', 0.1)
        assert sketch > 0 and sketch <= 1, f"sketch_eps should in (0, 1], got {sketch}"
        self.buckets = math.ceil(1.0 / sketch)
        self.seed = int(params.pop('seed', 42))

        assert len(params) == 0, f"Unknown params {list(params.keys())}"

    def _global_setup(self) -> None:
        buckets_maps = list()
        for worker in self.workers:
            m = worker.global_setup(self.x[worker.device].data, self.buckets, self.seed)
            buckets_maps.append(m.to(self.spu))

        self.spu_context = self.spu(split_fn.global_setup)(
            buckets_maps,
            self.y.to(self.spu),
            self.seed,
            self.reg_lambda,
            self.lr,
        )
        self.pred = self.spu(split_fn.init_pred, static_argnames=('base', 'samples'))(
            base=self.base, samples=self.samples
        )
        wait([self.spu_context, self.pred])

    def _tree_setup(self) -> None:
        col_buckets_choices = []
        works_buckets_count = []
        for pyu_work in self.workers:
            choices, count = pyu_work.tree_setup(self.colsample)
            works_buckets_count.append(count)
            if self.colsample < 1:
                # 1. column sample choices is generate by public param 'seed', choices is not a private value
                # 2. spu function need to use this choices to slicing array, this operation cannot be done in secret sharing.
                # SO, choices send to spu vis public.
                col_buckets_choices.append(
                    choices.to(self.spu, MoveConfig(spu_vis='public'))
                )

        for worker in self.workers:
            worker.update_buckets_count(
                [col.to(worker.device) for col in works_buckets_count]
            )

        self.spu_context = self.spu(
            split_fn.tree_setup, static_argnames=("objective", "samples", "subsample")
        )(
            self.spu_context,
            self.pred,
            col_buckets_choices,
            objective=self.obj,
            samples=self.samples,
            subsample=self.subsample,
        )

    def train(
        self,
        params: Dict,
        dtrain: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
    ) -> XgbModel:
        '''train on dtrain and label.

        Args:
            dtrain: {FedNdarray, VDataFrame}
                vertical split dataset.
            label: {FedNdarray, VDataFrame}
                label column.
            params: Dict
                booster params, details are as follows

        booster params details:

            num_boost_round : int, default=10
                Number of boosting iterations.
                range: [1, 1024]
            'max_depth': Maximum depth of a tree.
                default: 5
                range: [1, 16]
            'learning_rate': Step size shrinkage used in update to prevents overfitting.
                default: 0.3
                range: (0, 1]
            'objective': Specify the learning objective.
                default: 'logistic'
                range: ['linear', 'logistic']
            'reg_lambda': L2 regularization term on weights.
                default: 0.1
                range: [0, 10000]
            'subsample': Subsample ratio of the training instances.
                default: 1
                range: (0, 1]
            'colsample_bytree': Subsample ratio of columns when constructing each tree.
                default: 1
                range: (0, 1]
            'sketch_eps': This roughly translates into O(1 / sketch_eps) number of bins.
                default: 0.1
                range: (0, 1]
            'base_score': The initial prediction score of all instances, global bias.
                default: 0
            'seed': Pseudorandom number generator seed.
                default: 42

        Return:
            XgbModel
        '''
        start = time.time()
        self._prepare(params, dtrain, label)
        self._global_setup()
        logging.info(f"global_setup time {time.time() - start}s")

        model = XgbModel(self.spu, self.obj, self.base)
        while len(model.trees) < self.trees:
            start = time.time()

            self._tree_setup()

            tree, weight = self._train_tree()
            model.trees.append(tree)
            model.weights.append(weight)

            if len(model.trees) < self.trees:
                self._update_pred(tree, weight)
                wait([self.pred])
            else:
                wait(list(tree.values()) + [weight])

            logging.info(f"epoch {len(model.trees) - 1} time {time.time() - start}s")

        return model

    def _train_level(self, nodes_s: SPUObject, level: int) -> SPUObject:
        last_level = level == (self.depth - 1)

        spu_split_buckets, self.spu_context = self.spu(
            split_fn.find_best_split_bucket,
            static_argnames='last_level',
            num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
            user_specified_num_returns=2,
        )(self.spu_context, nodes_s, last_level=last_level)

        lchild_ss = []
        for worker in self.workers:
            # In the final tree model, which party hold the split feature for tree nodes is public information.
            # so, we can reveal 'split_buckets' to each pyu.
            lchild_s = worker.do_split(spu_split_buckets.to(worker.device))
            lchild_ss.append(lchild_s.to(self.spu))

        childs_s = self.spu(split_fn.get_child_select)(nodes_s, lchild_ss)

        return childs_s

    def _train_tree(self) -> Tuple[Dict[PYU, PYUObject], SPUObject]:
        root_s = self.spu(split_fn.root_select, static_argnames=('samples',))(
            samples=self.samples
        )

        nodes_s = root_s
        for level in range(self.depth + 1):
            if level < self.depth:
                # split nodes
                nodes_s = self._train_level(nodes_s, level)
            else:
                # leaf nodes
                weight = self.spu(split_fn.do_leaf)(self.spu_context, nodes_s)

        tree = {w.device: w.tree_finish() for w in self.workers}
        return tree, weight
