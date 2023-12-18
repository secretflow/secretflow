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
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from secretflow.device import PYUObject, reveal
from secretflow.ml.boost.sgb_v.core.params import default_params
from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

from ....core.distributed_tree.distributed_tree import DistributedTree
from ..bucket_sum_calculator import BucketSumCalculator
from ..component import Devices, print_params
from ..gradient_encryptor import GradientEncryptor
from ..leaf_manager import LeafManager
from ..logging import LoggingParams, LoggingTools
from ..loss_computer import LossComputer
from ..node_selector import NodeSelector
from ..order_map_manager import OrderMapManager
from ..sampler import Sampler
from ..shuffler import Shuffler
from ..split_finder import SplitFinder
from ..split_tree_builder import SplitTreeBuilder
from .tree_trainer import TreeTrainer


@dataclass
class LevelWiseTreeTrainerComponents:
    leaf_manager: LeafManager = LeafManager()
    node_selector: NodeSelector = NodeSelector()
    sampler: Sampler = Sampler()
    shuffler: Shuffler = Shuffler()
    gradient_encryptor: GradientEncryptor = GradientEncryptor()
    loss_computer: LossComputer = LossComputer()
    bucket_sum_calculator: BucketSumCalculator = BucketSumCalculator()
    split_finder: SplitFinder = SplitFinder()
    split_tree_builder: SplitTreeBuilder = SplitTreeBuilder()


@dataclass
class LevelWiseTreeTrainerParams:
    """params specifically belonged to level wise booster, not its components.

    'max_depth': int, maximum depth of a tree.
            default: 5
            range: [1, 16]
    """

    max_depth: int = default_params.max_depth


class LevelWiseTreeTrainer(TreeTrainer):
    def __init__(self) -> None:
        self.components = LevelWiseTreeTrainerComponents()
        self.params = LevelWiseTreeTrainerParams()
        self.logging_params = LoggingParams()

    def show_params(self):
        super().show_params()
        print_params(self.params)
        print_params(self.logging_params)

    def set_params(self, params: dict):
        super().set_params(params)
        self._set_trainer_params(params)

    def get_params(self, params: dict):
        super().get_params(params)
        self._get_trainer_params(params)

    def set_devices(self, devices: Devices):
        super().set_devices(devices)
        self.workers = devices.workers
        self.label_holder = devices.label_holder
        self.party_num = len(self.workers)

    def set_actors(self, actors: SGBActor):
        return super().set_actors(actors)

    def _get_trainer_params(self, params: dict):
        params['max_depth'] = self.params.max_depth
        LoggingTools.logging_params_write_dict(params, self.logging_params)

    def _set_trainer_params(self, params: dict):
        depth = params.get('max_depth', default_params.max_depth)

        self.params.max_depth = depth
        self.logging_params = LoggingTools.logging_params_from_dict(params)

    def train_tree_context_setup(
        self,
        cur_tree_num: int,
        order_map_manager: OrderMapManager,
        y: PYUObject,
        pred: Union[PYUObject, np.ndarray],
        sample_num: Union[PYUObject, int],
    ):
        logging.info("train tree context set up.")
        # reset caches
        self.components.split_tree_builder.reset()
        self.components.leaf_manager.clear_leaves()
        self.components.bucket_sum_calculator.reset_cache()
        logging.debug("cache resetted.")

        # sub sampling
        feature_buckets = order_map_manager.get_feature_buckets()
        col_choices, total_buckets = self.components.sampler.generate_col_choices(
            feature_buckets
        )
        self.components.split_tree_builder.set_col_choices_and_buckets(
            col_choices, total_buckets, feature_buckets
        )
        g, h = self.components.loss_computer.compute_gh(y, pred)
        row_choices, weight = self.components.sampler.generate_row_choices(
            sample_num, g
        )

        order_map = order_map_manager.get_order_map()
        self.bucket_lists = order_map_manager.get_bucket_lists(col_choices)
        self.order_map_sub = self.components.sampler.apply_v_fed_sampling(
            order_map, row_choices, col_choices
        )
        if self.components.sampler.should_row_subsampling():
            self.row_choices = reveal(row_choices)
        else:
            # Avoid transmission of None object in ic_mode
            self.row_choices = None

        self.node_select_shape = (
            1,
            reveal(sample_num) if self.row_choices is None else self.row_choices.size,
        )

        self.bucket_num = order_map_manager.buckets
        logging.debug("sub sampled (per tree).")

        # compute g, h and encryption
        g = self.components.sampler.apply_vector_sampling_weighted(
            g, row_choices, weight
        )
        h = self.components.sampler.apply_vector_sampling_weighted(
            h, row_choices, weight
        )
        self.components.loss_computer.compute_abs_sums(g, h)
        logging.debug("g h computed.")

        self.should_stop = reveal(self.components.loss_computer.check_early_stop())
        if self.should_stop:
            logging.debug("early stopped.")
            return
        logging.debug("not early stopped.")
        self.components.loss_computer.compute_scales()

        g, h = self.components.loss_computer.scale_gh(g, h)
        self.g = g
        self.h = h
        logging.debug("g h scaled.")

        gh = self.components.gradient_encryptor.pack(g, h)
        encrypted_gh = self.components.gradient_encryptor.encrypt(gh, cur_tree_num)
        self.encrypted_gh_dict = self.components.gradient_encryptor.cache_to_workers(
            encrypted_gh, gh
        )
        logging.debug("g h encrypted.")

    @LoggingTools.enable_logging
    def train_tree(
        self,
        cur_tree_num,
        order_map_manager: OrderMapManager,
        y: PYUObject,
        pred: Union[PYUObject, np.ndarray],
        sample_num: Union[PYUObject, int],
    ) -> DistributedTree:
        self.train_tree_context_setup(
            cur_tree_num, order_map_manager, y, pred, sample_num
        )
        if self.should_stop:
            return None
        logging.info("begin train tree.")
        row_num = self.node_select_shape[1]
        g, h = self.g, self.h
        root_select = self.components.node_selector.root_select(row_num)

        # level wise train begins
        split_node_selects = root_select
        split_node_indices = [0]
        logging.debug("beging level wise training.")
        for level in range(self.params.max_depth):
            logging.debug(f"training level {level}.")
            split_node_selects, split_node_indices = self._train_level(
                split_node_selects,
                split_node_indices,
                level,
                cur_tree_num,
                order_map_manager,
            )
            if reveal(self.components.node_selector.is_list_empty(split_node_indices)):
                # pruned all nodes
                break

        # leaf nodes
        # label_holder calc weights
        self.components.leaf_manager.extend_leaves(
            split_node_selects, split_node_indices
        )
        weight = self.components.leaf_manager.compute_leaf_weights(g, h)
        leaf_node_indices = self.components.leaf_manager.get_leaf_indices()
        tree = DistributedTree()
        tree.set_enable_packbits(
            self.components.bucket_sum_calculator.params.enable_packbits
        )
        self.components.split_tree_builder.insert_split_trees_into_distributed_tree(
            tree, leaf_node_indices
        )
        tree.set_leaf_weight(self.label_holder, weight)
        return tree

    @LoggingTools.enable_logging
    def _train_level(
        self,
        split_node_selects: PYUObject,
        split_node_indices: Union[List[int], PYUObject],
        level: int,
        tree_num: int,
        order_map_manager: OrderMapManager,
    ) -> Tuple[PYUObject, PYUObject, PYUObject, PYUObject]:
        last_level = level == (self.params.max_depth - 1)

        (
            label_holder_split_buckets,
            gain_is_cost_effective,
        ) = self._find_best_split_bucket(
            split_node_selects, last_level, tree_num, level
        )

        # split not in party will be marked as -1
        split_buckets_viewed_each_party = (
            self.components.split_tree_builder.split_bucket_to_partition(
                label_holder_split_buckets
            )
        )
        # -1 will retains
        unmasked_split_buckets_viewed_each_party = (
            self.components.shuffler.unshuffle_split_buckets(
                split_buckets_viewed_each_party
            )
        )
        split_feature_buckets_each_party = (
            self.components.split_tree_builder.get_split_feature_list_wise_each_party(
                unmasked_split_buckets_viewed_each_party
            )
        )
        left_selects_each_party = (
            order_map_manager.batch_compute_left_child_selects_each_party(
                split_feature_buckets_each_party, self.row_choices
            )
        )
        split_points = order_map_manager.batch_query_split_points_each_party(
            split_feature_buckets_each_party
        )
        select_shape = self.node_select_shape
        lchild_ss = self.components.split_tree_builder.do_split_list_wise_each_party(
            split_feature_buckets_each_party,
            split_points,
            left_selects_each_party,
            gain_is_cost_effective,
            split_node_indices,
            select_shape,
        )
        (
            childs_s,
            split_node_indices,
            pruned_s,
            pruned_node_indices,
        ) = self.components.node_selector.get_child_select(
            split_node_selects, lchild_ss, gain_is_cost_effective, split_node_indices
        )
        self.components.leaf_manager.extend_leaves(pruned_s, pruned_node_indices)
        return childs_s, split_node_indices

    def _find_best_split_bucket(
        self,
        split_node_selects: PYUObject,
        is_last_level: bool,
        tree_num: int,
        level: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute the gradient sums of the containing instances in each split bucket
        and find best split bucket for each node which has the max split gain.

        Args:
            split_node_selects: PYUObject. List[np.ndarray] at label_holder. Sample select indexes of each node from same tree level.
            last_level: bool. if this split is last level, next level is leaf nodes.
            tree_num: int. which tree is training
            level: int. which level is training

        Return:
            idx of split bucket for each node, and indicator if gain > gamma
        """

        # only compute the gradient sums of left or right children node. (choose fewer ones)
        (
            children_split_node_selects,
            is_lefts,
            node_num,
        ) = self.components.node_selector.pick_children_node_ss(split_node_selects)
        # all parties knows the shape of tree, and which nodes in them, so this is fine.
        is_lefts = reveal(is_lefts)

        (
            level_nodes_G,
            level_nodes_H,
        ) = self.components.bucket_sum_calculator.calculate_bucket_sum_level_wise(
            self.components.shuffler,
            self.encrypted_gh_dict,
            children_split_node_selects,
            is_lefts,
            self.order_map_sub,
            self.bucket_num,
            self.bucket_lists,
            self.components.gradient_encryptor,
            node_num,
            self.node_select_shape,
        )
        level_nodes_G, level_nodes_H = self.components.loss_computer.reverse_scale_gh(
            level_nodes_G, level_nodes_H
        )
        (
            split_buckets,
            gain_is_cost_effective,
        ) = self.components.split_finder.find_best_splits_level_wise(
            level_nodes_G, level_nodes_H, tree_num, level
        )
        # all parties including driver know the shape of tree in each node
        # hence all parties including driver will know the pruning results.
        # hence we can reveal gain_is_cost_effective
        gain_is_cost_effective = reveal(gain_is_cost_effective)
        self.components.bucket_sum_calculator.update_level_cache(
            is_last_level, gain_is_cost_effective
        )

        return split_buckets, gain_is_cost_effective
