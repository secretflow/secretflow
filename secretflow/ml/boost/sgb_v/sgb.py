# Copyright 2023 Ant Group Co., Ltd.
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

import logging
import math
import pickle
import time
from typing import Dict, List, Tuple, Union

import numpy as np
from heu import phe

from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device import HEU, PYU, PYUObject, reveal, wait
from secretflow.device.device.heu import HEUMoveConfig

from .core.cache.level_cache import LevelCache
from .core.distributed_tree.distributed_tree import DistributedTree
from .core.label_holder.label_holder import LabelHolder
from .core.preprocessing.params import LabelHolderInfo
from .core.preprocessing.preprocessing import prepare_dataset, validate_sgb_params_dict
from .core.split_tree_trainer.split_tree_trainer import SplitTreeTrainer as Worker
from .model import SgbModel

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)


class Sgb:
    """
    This class provides both classification and regression tree boosting (also known as GBDT, GBM)
    for vertical split dataset setting by using secure boost.

    SGB is short for SecureBoost. Compared to its safer counterpart SS-XGB, SecureBoost focused on protecting label holder.

    Args:
        heu: secret device running homomorphic encryptions

    """

    def __init__(self, heu: HEU) -> None:
        # todo: distributed SGB, work with multiple heu to support large dataset.
        self.heu = heu

    def _prepare(
        self,
        params: Dict,
        dataset: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
        audit_paths: Dict = {},
    ) -> None:
        x, x_shape = prepare_dataset(dataset)
        y, y_shape = prepare_dataset(label)
        assert len(x_shape) == 2, "only support 2D-array on dtrain"
        assert len(y_shape) == 1 or y_shape[1] == 1, "label only support one label col"
        self.samples = y_shape[0]
        assert self.samples == x_shape[0], "dtrain & label are not aligned"
        assert len(y.partitions) == 1, "label only support one partition"
        label_holder = [*y.partitions.keys()][0]
        assert (
            label_holder.party == self.heu.sk_keeper_name()
        ), f"HEU sk keeper party {self.heu.sk_keeper_name()}, mismatch with label_holder device's party {label_holder.party}"
        # determine label_holder from label holder
        self.label_holder = label_holder

        self.y = list(y.partitions.values())[0]
        self.workers = [Worker(idx, device=pyu) for idx, pyu in enumerate(x.partitions)]
        self.x = x.partitions

        validated_params = validate_sgb_params_dict(params)

        self.subsample = validated_params.subsample
        self.seed = validated_params.seed
        self.obj = validated_params.objective
        self.base = validated_params.base_score
        self.trees = validated_params.num_boost_round
        self.depth = validated_params.max_depth
        self.colsample = validated_params.colsample_by_tree
        fxp_scale = np.power(2, validated_params.fixed_point_parameter)
        self.buckets = math.ceil(1.0 / validated_params.sketch_eps)
        self.gh_encoder = phe.BatchFloatEncoderParams(scale=fxp_scale)

        self.audit_paths = audit_paths

        if self.label_holder.party in self.audit_paths:
            path = self.audit_paths[self.label_holder.party] + ".gamma.pickle"
            write_log(validated_params.gamma, path)

        # pack label holder info
        self.label_holder_info = LabelHolderInfo(
            self.seed,
            validated_params.reg_lambda,
            validated_params.gamma,
            validated_params.learning_rate,
            validated_params.base_score,
            self.samples,
            self.subsample,
            validated_params.objective,
        )

        assert len(params) == 0, f"Unknown params {list(params.keys())}"

    def _global_setup(self) -> None:
        self.order_map = FedNdarray(
            {
                worker.device: worker.global_setup(
                    self.x[worker.device].data, self.buckets, self.seed
                )
                for worker in self.workers
            },
            partition_way=PartitionWay.VERTICAL,
        )

        self.train_label_holder = LabelHolder(
            self.label_holder_info, device=self.label_holder
        )
        self.train_label_holder.set_y(self.y)
        self.pred = self.train_label_holder.init_pred()

        self.worker_caches = {
            worker: LevelCache()
            for worker in self.workers
            if worker.device != self.label_holder
        }

    def _tree_setup(self, tree_num) -> None:
        col_choices = {}
        works_buckets_count = []
        for pyu_work in self.workers:
            choices, count = pyu_work.tree_setup(self.colsample)
            works_buckets_count.append(count)
            if self.colsample < 1:
                # 1. column sample choices is generate by public param 'seed', choices is not a private value
                col_choices[pyu_work.device] = choices

        self.col_choices = FedNdarray(
            partitions=col_choices, partition_way=PartitionWay.VERTICAL
        )

        for worker in self.workers:
            worker.set_buckets_count(
                [col.to(worker.device) for col in works_buckets_count]
            )

        self.train_label_holder.setup_context(self.pred)

        def move_config(pyu, edr):
            move_config = HEUMoveConfig()
            move_config.heu_encoder = edr
            move_config.heu_dest_party = pyu.party
            return move_config

        # make encrypted gh here

        if self.label_holder.party in self.audit_paths:
            path = self.audit_paths[self.label_holder.party] + ".tree_" + str(tree_num)
        else:
            path = None

        encypted_gh = (
            self.train_label_holder.get_gh()
            .to(self.heu, move_config(self.label_holder, self.gh_encoder))
            .encrypt(path)
        )
        # encrypt once, send once.
        self.encrypted_gh = {
            worker.device: encypted_gh.to(
                self.heu, move_config(worker.device, self.gh_encoder)
            )
            for worker in self.workers
            if worker.device != self.label_holder
        }
        wait([self.train_label_holder.get_gh(), *self.encrypted_gh.values()])
        self._sub_sampling()

    def train(
        self,
        params: Dict,
        dtrain: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
        audit_paths: Dict = {},
    ) -> SgbModel:
        """train on dtrain and label.

        Args:
            params: Dict
                booster params, details are as follows
            dtrain: {FedNdarray, VDataFrame}
                vertical split dataset.
            label: {FedNdarray, VDataFrame}
                label column.
            audit_paths: {party: party_audit_path} for each party.
                party_audit_path is a file location for gradients.
                Leave it empty if you do not need audit function.

        booster params details:
            num_boost_round : int, default=10
                Number of boosting iterations.
                range: [1, 1024]
            'max_depth': int, maximum depth of a tree.
                default: 5
                range: [1, 16]
            'learning_rate': float, step size shrinkage used in update to prevent overfitting.
                default: 0.3
                range: (0, 1]
            'objective': Specify the learning objective.
                default: 'logistic'
                range: ['linear', 'logistic']
            'reg_lambda': float. L2 regularization term on weights.
                default: 0.1
                range: [0, 10000]
            'gamma': float. Greater than 0 means pre-pruning enabled.
                Gain less than it will not induce split node.
                default: 0.1
                range: [0, 10000]
            'subsample': Subsample ratio of the training instances.
                default: 1
                range: (0, 1]
            'colsample_by_tree': Subsample ratio of columns when constructing each tree.
                default: 1
                range: (0, 1]
            'sketch_eps': This roughly translates into O(1 / sketch_eps) number of bins.
                default: 0.1
                range: (0, 1]
            'base_score': The initial prediction score of all instances, global bias.
                default: 0
            'seed': Pseudorandom number generator seed.
                default: 42
            'fixed_point_parameter': int. Any floating point number encoded by heu,
             will multiply a scale and take the round,
             scale = 2 ** fixed_point_parameter.
             larger value may mean more numerical accurate,
             but too large will lead to overflow problem.
             See HEU's document for more details.
                default: 20

        Return:
            SgbModel
        """
        start = time.perf_counter()
        self._prepare(params, dtrain, label, audit_paths)
        self._global_setup()
        logging.info(f"global_setup time {time.perf_counter() - start}s")

        model = SgbModel(self.label_holder, self.obj, self.base)
        while len(model.trees) < self.trees:
            start = time.perf_counter()
            cur_tree_num = len(model.trees)
            self._tree_setup(cur_tree_num)

            tree = self._train_tree(cur_tree_num)
            model._insert_distributed_tree(tree)
            cur_tree_num = len(model.trees)

            if cur_tree_num < self.trees:
                prev_pred = self.pred
                self.pred = self.label_holder(lambda x, y: x + y)(
                    prev_pred, tree.predict(self.x)
                )
                wait([self.pred])
            else:
                wait(tree)

            logging.info(
                f"epoch {cur_tree_num - 1} time {time.perf_counter() - start}s"
            )

        return model

    def _train_tree(self, tree_num: int) -> DistributedTree:
        self.train_label_holder.clear_leaves()
        root_select = self.train_label_holder.root_select()

        split_node_selects = root_select
        split_node_indices = [0]
        for level in range(self.depth):
            split_node_selects, split_node_indices = self._train_level(
                split_node_selects, split_node_indices, level, tree_num
            )
            if reveal(self.train_label_holder.is_list_empty(split_node_indices)):
                # pruned all nodes
                break

        # leaf nodes
        # label_holder calc weights
        self.train_label_holder.extend_leaves(split_node_selects, split_node_indices)
        weight = self.train_label_holder.do_leaf()
        leaf_node_indices = self.train_label_holder.get_leaf_indices()
        tree = DistributedTree()
        for w in self.workers:
            tree.insert_split_tree(
                w.device, w.tree_finish(leaf_node_indices.to(w.device))
            )
        tree.set_leaf_weight(self.label_holder, weight)
        return tree

    def _train_level(
        self,
        split_node_selects: PYUObject,
        split_node_indices: Union[List[int], PYUObject],
        level: int,
        tree_num: int,
    ) -> Tuple[PYUObject, PYUObject, PYUObject, PYUObject]:
        last_level = level == (self.depth - 1)
        (
            label_holder_split_buckets,
            gain_is_cost_effective,
        ) = self._find_best_split_bucket(split_node_selects, last_level)
        if self.label_holder.party in self.audit_paths:
            split_info_path = (
                self.audit_paths[self.label_holder.party]
                + ".split_buckets.tree_"
                + str(tree_num)
                + ".level_"
                + str(level)
                + ".pickle"
            )

            self.label_holder(write_log)(label_holder_split_buckets, split_info_path)

        def compute_child_selects(worker):
            sampled_rows = (
                self.train_label_holder.get_sub_choices().to(worker.device)
                if self.subsample < 1
                else None
            )

            split_node_indices_here = (
                split_node_indices.to(worker.device)
                if isinstance(split_node_indices, PYUObject)
                else split_node_indices
            )

            return worker.do_split(
                label_holder_split_buckets.to(worker.device),
                sampled_rows,
                gain_is_cost_effective,
                split_node_indices_here,
            )

        # In the final tree model, which party hold the split feature for tree nodes is public information.
        # so, we can reveal 'split_buckets' to each pyu.
        lchild_ss = [
            compute_child_selects(worker).to(self.label_holder)
            for worker in self.workers
        ]
        (
            childs_s,
            split_node_indices,
            pruned_s,
            pruned_node_indices,
        ) = self.train_label_holder.get_child_select(
            split_node_selects, lchild_ss, gain_is_cost_effective, split_node_indices
        )
        self.train_label_holder.extend_leaves(pruned_s, pruned_node_indices)
        return childs_s, split_node_indices

    def _find_best_split_bucket(
        self, split_node_selects: PYUObject, last_level: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute the gradient sums of the containing instances in each split bucket
        and find best split bucket for each node which has the max split gain.

        Args:
            split_node_selects: PYUObject. List[np.ndarray] at label_holder. Sample select indexes of each node from same tree level.
            last_level: bool. if this split is last level, next level is leaf nodes.

        Return:
            idx of split bucket for each node, and indicator if gain > gamma
        """

        # only compute the gradient sums of left or right children node. (choose fewer ones)
        (
            children_split_node_selects,
            is_lefts,
        ) = self.train_label_holder.pick_children_node_ss(split_node_selects)
        # all parties knows the shape of tree, and which nodes in them, so this is fine.
        is_lefts = reveal(is_lefts)

        bucket_sums_list = [[] for _ in range(len(self.workers))]
        for i, worker in enumerate(self.workers):
            if worker.device != self.label_holder:
                self.worker_caches[worker].reset_level_nodes_GH()
                worker.reset_shuffle_mask()
                bucket_sums = self.encrypted_gh[
                    worker.device
                ].batch_feature_wise_bucket_sum(
                    children_split_node_selects,
                    self.order_map_sub.partitions[worker.device],
                    self.buckets + 1,
                    True,
                )

                self.worker_caches[worker].collect_level_node_GH_level_wise(
                    bucket_sums, is_lefts
                )

                bucket_sums = self.worker_caches[worker].get_level_nodes_GH()
                bucket_sums = [
                    bucket_sum[worker.create_shuffle_mask(j)]
                    for j, bucket_sum in enumerate(bucket_sums)
                ]

                bucket_sums_list[i] = [
                    bucket_sum.to(
                        self.label_holder,
                        move_config(self.label_holder, self.gh_encoder),
                    )
                    for bucket_sum in bucket_sums
                ]

            else:
                self.train_label_holder.reset_level_nodes_GH()
                bucket_sums = self.train_label_holder.batch_select_sum(
                    self.train_label_holder.get_gh(),
                    children_split_node_selects,
                    self.order_map_sub.partitions[worker.device],
                    self.buckets + 1,
                )
                self.train_label_holder.collect_level_node_GH_level_wise(
                    bucket_sums, is_lefts
                )
                bucket_sums = self.train_label_holder.get_level_nodes_GH()
                bucket_sums_list[i] = bucket_sums

        self.train_label_holder.regroup_and_collect_level_nodes_GH(bucket_sums_list)

        (
            split_buckets,
            gain_is_cost_effective,
        ) = self.train_label_holder.find_best_splits()
        # all parties including driver know the shape of tree in each node
        # hence all parties including driver will know the pruning results.
        # hence we can reveal gain_is_cost_effective
        gain_is_cost_effective = reveal(gain_is_cost_effective)
        [
            self.worker_caches[worker].update_level_cache(
                last_level, gain_is_cost_effective
            )
            for worker in self.workers
            if worker.device != self.label_holder
        ]
        self.train_label_holder.update_level_cache(last_level, gain_is_cost_effective)
        return split_buckets, gain_is_cost_effective

    def _sub_sampling(self):
        self.order_map_sub = self.order_map
        # sample cols and rows of bucket_map
        if len(self.col_choices.partitions.keys()) > 0 and self.subsample < 1:
            # sub choices is stored in context owned by label_holder and shared to all workers.
            self.order_map_sub = FedNdarray(
                partitions={
                    pyu: pyu(lambda x, y, z: x[y, :][:, z])(
                        partition,
                        self.train_label_holder.get_sub_choices().to(pyu),
                        self.col_choices.partitions[pyu],
                    )
                    for pyu, partition in self.order_map_sub.partitions.items()
                },
                partition_way=PartitionWay.VERTICAL,
            )
        # only sample cols
        elif len(self.col_choices.partitions) > 0:
            self.order_map_sub = FedNdarray(
                partitions={
                    pyu: pyu(lambda x, y: x[:, y])(
                        partition, self.col_choices.partitions[pyu]
                    )
                    for pyu, partition in self.order_map_sub.partitions.items()
                },
                partition_way=PartitionWay.VERTICAL,
            )
        # only sample rows
        elif self.subsample < 1:
            self.order_map_sub = FedNdarray(
                partitions={
                    pyu: pyu(lambda x, y: x[y, :])(
                        partition,
                        self.train_label_holder.get_sub_choices().to(pyu),
                    )
                    for pyu, partition in self.order_map_sub.partitions.items()
                },
                partition_way=PartitionWay.VERTICAL,
            )
        self.order_map_sub_shapes = self.order_map_sub.partition_shape()


def move_config(pyu, params):
    move_config = HEUMoveConfig()
    if isinstance(pyu, PYU):
        move_config.heu_encoder = params
        move_config.heu_dest_party = pyu.party
    return move_config


def write_log(x, path):
    with open(path, "wb") as f:
        pickle.dump(x, f)
