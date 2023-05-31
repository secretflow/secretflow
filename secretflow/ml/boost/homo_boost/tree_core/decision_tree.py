#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


import json
import logging
from typing import List

import numpy as np
import pandas
import xgboost as xgb

from secretflow.ml.boost.homo_boost.tree_core.feature_histogram import FeatureHistogram
from secretflow.ml.boost.homo_boost.tree_core.feature_importance import (
    FeatureImportance,
)
from secretflow.ml.boost.homo_boost.tree_core.node import Node
from secretflow.ml.boost.homo_boost.tree_core.splitter import SplitInfo, Splitter
from secretflow.ml.boost.homo_boost.tree_param import TreeParam


class DecisionTree(object):
    """Class for local version decision tree

    Attributes:
        tree_param: params for tree build
        data: training data, HdataFrame
        bin_split_points: global binning infos
        tree_id: tree id
        group_id: group id indicates which class the tree classifies
        iter_round: iteration round
        hess_key: unique column name for hess value
        grad_key: unique column name for grad value
    """

    def __init__(
        self,
        tree_param: TreeParam = None,
        data: pandas.DataFrame = None,
        bin_split_points: np.ndarray = None,
        tree_id: int = None,
        group_id: int = None,
        iter_round: int = None,
        grad_key: str = "grad",
        hess_key: str = "hess",
        label_key: str = "label",
    ):
        # input parameters
        self.criterion_method = tree_param.criterion_method
        self.criterion_params = [
            tree_param.reg_lambda,
            tree_param.reg_alpha,
            tree_param.decimal,
        ]
        self.max_depth = tree_param.max_depth
        self.min_sample_split = tree_param.min_sample_split
        self.min_impurity_split = tree_param.gamma
        self.min_leaf_node = tree_param.min_leaf_node
        self.max_split_nodes = tree_param.max_split_nodes
        self.feature_importance_type = tree_param.importance_type
        self.objective = tree_param.objective
        self.learning_rate = tree_param.eta
        self.use_missing = tree_param.use_missing
        self.min_child_weight = tree_param.min_child_weight
        self.num_class = tree_param.num_class
        # col subsample
        self.random_state = tree_param.random_state
        self.colsample_bytree = tree_param.colsample_bytree
        self.colsample_by_level = tree_param.colsample_byleval
        # runtime variable
        self.feature_importance = {}
        self.tree_node = []
        self.cur_layer_nodes = []
        self.cur_layer_datas = []
        self.cur_to_split_nodes = []
        self.tree_node_num = 0
        self.num_parallel = tree_param.num_parallel
        self.splitter = Splitter(
            self.criterion_method,
            self.criterion_params,
            self.min_impurity_split,
            self.min_sample_split,
            self.min_leaf_node,
            self.min_child_weight,
        )  # splitter for finding splits

        self.bin_split_points = bin_split_points
        self.valid_features = None
        # histogram
        self.hist_computer = FeatureHistogram()

        # tree idx
        self.tree_id = tree_id
        self.group_id = group_id
        self.iter_round = iter_round

        # for training
        self.hess_key = hess_key
        self.grad_key = grad_key
        self.label_key = label_key
        self.xgb_version = list(map(int, xgb.__version__.split(".")))
        # data
        if data is not None:
            self.data = data
            self.header = data.columns.tolist()
            self.columns_filter()

    def feature_col_sample(self, all_features: List[str], sample_rate: float = 1.0):
        """Column sample for features
        Args:
            all_features: A list of feature names for all columns
            sample_rate: subsample rate, a float-number in [0, 1]
        Returns:
            valid_features: A dict of valid features, which will be use in this round built
        """
        assert (
            sample_rate <= 1
        ), f"sample_rate must be less than or equal to 1, but got {sample_rate}"
        valid_features = {}
        all_feature_count = len(all_features)
        sampled_feature_count = round(all_feature_count * sample_rate)
        # fix seed to generate same col sample
        np.random.seed(self.random_state * (1 + self.tree_id))
        sampled_feature_idx = np.random.choice(
            all_feature_count, sampled_feature_count, replace=False
        )
        for idx in range(all_feature_count):
            if idx in sampled_feature_idx:
                valid_features[idx] = True
            else:
                valid_features[idx] = False
        return valid_features

    def get_feature_importance(self):
        return self.feature_importance

    def convert_bin_to_real(self):
        """convert bid to real value"""
        for node in self.tree_node:
            if not node.is_leaf:
                node.bid = self.bin_split_points[node.fid][node.bid]

    def columns_filter(self):
        if self.hess_key in self.header:
            self.header.remove(self.hess_key)
        if self.grad_key in self.header:
            self.header.remove(self.grad_key)
        if self.label_key in self.header:
            self.header.remove(self.label_key)

    def get_grad_hess_sum(self, data_frame):
        """calculate sum of grad and hess
        Args:
            data_frame:data frame which contains hess and grad
        Returns:
            grad: sum of grad
            hess: sum of hess
        """
        sum_grad = data_frame[self.grad_key].sum()
        sum_hess = data_frame[self.hess_key].sum()
        return sum_grad, sum_hess

    def update_feature_importance(self, split_info):
        """Calculate feature importance
        default split count
        Args:
            split_info: Global optimal splitting information calculated from histogram
        """
        inc_split, inc_gain = 1, split_info.gain

        fid = split_info.best_fid

        if fid not in self.feature_importance:
            self.feature_importance[fid] = FeatureImportance(
                0, 0, self.feature_importance_type
            )

        self.feature_importance[fid].add_split(inc_split)
        if inc_gain is not None:
            self.feature_importance[fid].add_gain(inc_gain)

    def fit(self):
        """Entrance for local decision tree"""
        logging.debug(
            'begin to fit local decision tree, tree idx {}'.format(self.tree_id)
        )
        self.valid_features = self.feature_col_sample(
            self.header, self.colsample_bytree
        )
        # compute local g_sum and h_sum
        g_sum, h_sum = self.get_grad_hess_sum(self.data)

        # initialize node
        root_node = Node(
            id=0,
            sum_grad=g_sum,
            sum_hess=h_sum,
            weight=self.splitter.node_weight(g_sum, h_sum),
            sample_num=len(self.data),
        )

        self.cur_layer_node = [root_node]
        self.cur_layer_datas = [self.data]

        tree_height = self.max_depth + 1  # non-leaf node height + 1 layer leaf
        for dep in range(tree_height):
            if self.colsample_by_level < 1:
                self.valid_features = self.feature_col_sample(
                    self.header, self.colsample_by_level
                )
            if dep + 1 == tree_height:
                for node in self.cur_layer_node:
                    node.is_leaf = True
                    self.tree_node.append(node)
                break

            logging.debug(f'start to fit layer {dep}')

            agg_histograms = []
            for batch_id, i in enumerate(
                range(0, len(self.cur_layer_node), self.max_split_nodes)
            ):
                cur_to_split = self.cur_layer_node[i : i + self.max_split_nodes]
                cur_data_frame = self.cur_layer_datas[i : i + self.max_split_nodes]
                assert len(cur_to_split) == len(
                    cur_data_frame
                ), "node_to_split and data_frame_list must be aligned"
                logging.debug(
                    'computing histogram for batch{} at depth{}'.format(batch_id, dep)
                )
                local_histograms = self.hist_computer.calculate_histogram(
                    data_frame_list=cur_data_frame,
                    bin_split_points=self.bin_split_points,
                    valid_features=self.valid_features,
                    use_missing=self.use_missing,
                    grad_key=self.grad_key,
                    hess_key=self.hess_key,
                )

                agg_histograms += local_histograms
            split_info_list = self.splitter.find_split(
                agg_histograms, self.valid_features, self.use_missing
            )
            logging.debug('got best splits from arbiter')

            new_layer_node, new_layer_data = self.update_tree(
                self.cur_layer_node, split_info_list, self.cur_layer_datas
            )

            self.cur_layer_node = new_layer_node
            self.cur_layer_datas = new_layer_data

        self.convert_bin_to_real()

        logging.debug('fitting tree done')

    def update_tree(
        self,
        cur_to_split: List[Node],
        split_info: List[SplitInfo],
        cur_data_frames: List[pandas.DataFrame],
    ):
        """Tree update function
        Args:
            cur_to_split: List of nodes to be split
            split_info: Global optim split info
            cur_data_frames: List of dataframe in each node
        Returns:
            next_layer_node: List of nodes to be evaluated in the next iteration
            next_layer_data: List of data to be evaluated in the next iteration

        """
        logging.debug(
            'updating tree_node, cur layer has {} node'.format(len(cur_to_split))
        )
        next_layer_node, next_layer_data = [], []

        assert len(cur_to_split) == len(
            split_info
        ), "Num of nodes and split_info must have same length"

        for idx in range(len(cur_to_split)):
            if (
                split_info[idx].best_fid is None
                or split_info[idx].gain <= self.min_impurity_split
            ):
                cur_to_split[idx].is_leaf = True
                self.tree_node.append(cur_to_split[idx])
                continue

            cur_data_frame = cur_data_frames[idx]

            best_split_col = self.header[split_info[idx].best_fid]
            best_split_bin = self.bin_split_points[split_info[idx].best_fid][
                split_info[idx].best_bid
            ]

            sum_grad = cur_to_split[idx].sum_grad
            sum_hess = cur_to_split[idx].sum_hess

            cur_to_split[idx].fid = split_info[idx].best_fid
            cur_to_split[idx].bid = split_info[idx].best_bid
            cur_to_split[idx].missing_dir = split_info[idx].missing_dir

            p_id = cur_to_split[idx].id
            l_id, r_id = self.tree_node_num + 1, self.tree_node_num + 2
            cur_to_split[idx].left_nodeid, cur_to_split[idx].right_nodeid = l_id, r_id
            self.tree_node_num += 2

            l_g, l_h = split_info[idx].sum_grad, split_info[idx].sum_hess
            # create new left node and new right node
            left_data = cur_data_frame[cur_data_frame[best_split_col] < best_split_bin]
            left_node = Node(
                id=l_id,
                sum_grad=l_g,
                sum_hess=l_h,
                weight=self.splitter.node_weight(l_g, l_h) * self.learning_rate,
                parent_nodeid=p_id,
                sibling_nodeid=r_id,
                is_left_node=True,
                sample_num=len(left_data),
            )
            right_data = cur_data_frame[
                cur_data_frame[best_split_col] >= best_split_bin
            ]
            right_node = Node(
                id=r_id,
                sum_grad=sum_grad - l_g,
                sum_hess=sum_hess - l_h,
                weight=self.splitter.node_weight(sum_grad - l_g, sum_hess - l_h)
                * self.learning_rate,
                parent_nodeid=p_id,
                sibling_nodeid=l_id,
                is_left_node=False,
                sample_num=len(right_data),
            )

            next_layer_node.append(left_node)
            next_layer_data.append(left_data)

            next_layer_node.append(right_node)
            next_layer_data.append(right_data)
            cur_to_split[idx].loss_change = split_info[idx].gain
            self.tree_node.append(cur_to_split[idx])

            self.update_feature_importance(split_info[idx])

        return next_layer_node, next_layer_data

    def init_xgboost_model(self, model_path: str):
        """Init standard xgboost model
        Args:
            model_path: model path
        """
        model = {}

        json_objection = {}
        if self.objective == "reg:squarederror" or self.objective == "":
            json_objection["objective"] = {
                "name": self.objective,
                "reg_loss_param": {"scale_pos_weight": "1"},
            }
        elif self.objective == "binary:logistic" or self.objective == "reg:logistic":
            json_objection["objective"] = {
                "name": self.objective,
                "reg_loss_param": {"scale_pos_weight": "1"},
            }
        elif self.objective == "multi:softmax" or self.objective == "multi:softprob":
            json_objection["objective"] = {
                "name": self.objective,
                "softmax_multiclass_param": {"num_class": str(self.num_class)},
            }
        else:
            raise Exception(f"Unknow objection:{self.objective}")

        model["learner"] = {
            "attributes": {},
            "feature_names": self.header,
            "feature_types": ['float' for i in self.header],
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {"num_trees": 0, "size_leaf_vector": 0},
                    "tree_info": [],
                    "trees": [],
                },
                "name": "gbtree",
            },
            "learner_model_param": {
                "base_score": "5E-1",
                "num_class": str(self.num_class),
                "num_feature": str(len(self.header)),
            },
        }
        model["learner"]["objective"] = json_objection["objective"]
        model["version"] = self.xgb_version

        with open(model_path, "w") as dump_f:
            json.dump(model, dump_f)

    def save_xgboost_model(self, model_path: str, tree_nodes: List[Node]):
        """Transform tree info to standard xgboost model
        ref: https://xgboost.readthedocs.io/en/latest/dev/structxgboost_1_1TreeParam.html#aab8ff286e59f1bbab47bfa865da4a107
        Args:
            model_path: model path
            tree_nodes: federate decision tree internal model
        Returns:
            update standard xgboost model on the model path
        """
        with open(model_path, 'r') as load_f:
            json_model = json.load(load_f)
        tree_param = {
            "base_weights": [],
            "categories": [],
            "categories_nodes": [],
            "categories_segments": [],
            "categories_sizes": [],
            "default_left": [],
            "id": self.tree_id,
            "left_children": [],
            "loss_changes": [],
            "parents": [],
            "right_children": [],
            "split_conditions": [],
            "split_indices": [],
            "split_type": [],
            "sum_hessian": [],
            "tree_param": {
                "num_deleted": "0",
                "num_feature": str(len(self.header)),
                "num_nodes": str(len(tree_nodes)),
                "size_leaf_vector": "0",
            },
        }
        for node in tree_nodes:
            tree_param["base_weights"].append(
                node.weight if node.weight is not None else 0e0
            )
            tree_param["default_left"].append(True if node.missing_dir == -1 else False)
            tree_param["left_children"].append(
                node.left_nodeid if node.left_nodeid is not None else -1
            )
            tree_param["loss_changes"].append(node.loss_change)
            tree_param["parents"].append(
                node.parent_nodeid if node.parent_nodeid is not None else -1
            )
            tree_param["right_children"].append(
                node.right_nodeid if node.right_nodeid is not None else -1
            )
            tree_param["split_conditions"].append(
                node.bid if node.bid is not None else node.weight
            )
            tree_param["split_indices"].append(node.fid if node.fid is not None else 0)
            tree_param["split_type"].append(0)
            tree_param["sum_hessian"].append(node.sum_hess)

        json_model["learner"]["attributes"]["best_iteration"] = str(self.iter_round)
        json_model["learner"]["attributes"]["best_ntree_limit"] = str(
            self.iter_round + 1
        )

        json_model["learner"]["gradient_booster"]["model"]["tree_info"].append(
            self.group_id
        )
        json_model["learner"]["gradient_booster"]["model"]["trees"].append(tree_param)
        json_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ] = str(
            int(
                json_model["learner"]["gradient_booster"]["model"][
                    "gbtree_model_param"
                ]["num_trees"]
            )
            + 1
        )
        json_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "size_leaf_vector"
        ] = str(
            json_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "size_leaf_vector"
            ]
        )
        with open(model_path, "w") as dump_f:
            json.dump(json_model, dump_f, ensure_ascii=False)
