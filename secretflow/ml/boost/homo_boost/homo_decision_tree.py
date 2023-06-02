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


""" Homo Decision Tree """
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import numpy as np
import pandas as pd
import secretflow.device.link as link
from secretflow.ml.boost.homo_boost.tree_core.decision_tree import DecisionTree
from secretflow.ml.boost.homo_boost.tree_core.feature_histogram import (
    FeatureHistogram,
    HistogramBag,
)
from secretflow.ml.boost.homo_boost.tree_core.node import Node
from secretflow.ml.boost.homo_boost.tree_param import TreeParam


class HomoDecisionTree(DecisionTree):
    """Class for federated version decision tree

    Attributes:
        tree_param: params for tree build
        data: training data, HdataFrame
        bin_split_points: global binning infos
        tree_id: tree id
        group_id: group_id
        iter_round: iter_round in the total XGBoost training progress
        hess_key: unique column name for hess value
        grad_key: unique column name for grad value
        label_key: unique column name for label key

    """

    def __init__(
        self,
        tree_param: TreeParam = None,
        data: pd.DataFrame = None,
        bin_split_points: np.ndarray = None,
        group_id: int = None,
        tree_id: int = None,
        iter_round: int = None,
        hess_key: str = "hess",
        grad_key: str = "grad",
        label_key: str = "label",
    ):
        super(HomoDecisionTree, self).__init__(
            tree_param, grad_key=grad_key, hess_key=hess_key, label_key=label_key
        )

        self.data = data

        self.bin_split_points = bin_split_points
        self.group_id = group_id
        self.tree_id = tree_id
        self.iter_round = iter_round

        self._exist_key = {}
        self._sync_version = 0
        self.role = link.get_role()
        self.hess_key = hess_key
        self.grad_key = grad_key

    def key(self, name: str) -> str:
        if name in self._exist_key:
            key = self._exist_key[name]
        else:
            key = f"HomoDT/{self.group_id}/{self.tree_id}/{name}"
            self._exist_key[name] = key
        return key

    def get_valid_features_by_tree(self):
        if self.role == link.SERVER:
            self.header = [field for field in self.data.columns]
            self.columns_filter()
            self.valid_features = self.feature_col_sample(
                self.header, self.colsample_bytree
            )
            link.send_to_clients(
                name=self.key('valid_features_bytree'),
                value=self.valid_features,
                version=self._sync_version,
            )

        if self.role == link.CLIENT:
            self.header = [field for field in self.data.columns]
            self.columns_filter()
            self.valid_features = link.recv_from_server(
                name=self.key('valid_features_bytree'),
                version=self._sync_version,
            )
            self.feature_col_sample(self.header, self.colsample_bytree)

    def get_valid_features_by_level(self):
        if self.role == link.SERVER:
            self.valid_features = self.feature_col_sample(
                self.header, self.colsample_by_level
            )
            link.send_to_clients(
                name=self.key('valid_features_bylevel'),
                value=self.valid_features,
                version=self._sync_version,
            )

        if self.role == link.CLIENT:
            self.valid_features = link.recv_from_server(
                name=self.key('valid_features_bylevel'),
                version=self._sync_version,
            )

    def cal_root_node(self):
        if self.role == link.CLIENT:
            g_sum, h_sum = self.get_grad_hess_sum(self.data)
            # initialize node
            link.send_to_server(
                name=self.key('root_g_sum'), value=g_sum, version=self._sync_version
            )
            link.send_to_server(
                name=self.key('root_h_sum'), value=h_sum, version=self._sync_version
            )

        if self.role == link.SERVER:
            g_list = link.recv_from_clients(
                name=self.key('root_g_sum'),
                version=self._sync_version,
            )
            h_list = link.recv_from_clients(
                name=self.key('root_h_sum'),
                version=self._sync_version,
            )
            global_g_sum = np.sum(np.array(g_list), axis=0)
            global_h_sum = np.sum(np.array(h_list), axis=0)
            link.send_to_clients(
                name=self.key('global_root_g_sum'),
                value=global_g_sum,
                version=self._sync_version,
            )
            link.send_to_clients(
                name=self.key('global_root_h_sum'),
                value=global_h_sum,
                version=self._sync_version,
            )

        if self.role == link.CLIENT:
            g_sum = link.recv_from_server(
                name=self.key('global_root_g_sum'),
                version=self._sync_version,
            )
            h_sum = link.recv_from_server(
                name=self.key('global_root_h_sum'),
                version=self._sync_version,
            )
            root_node = Node(
                id=0,
                sum_grad=g_sum,
                sum_hess=h_sum,
                weight=self.splitter.node_weight(g_sum, h_sum),
                sample_num=len(self.data),
            )
            self.cur_layer_node = [root_node]
            self.cur_layer_datas = [self.data]

    @staticmethod
    def cal_local_hist_bags(
        cur_to_split,
        cur_data_frame,
        bin_split_points,
        valid_features,
        use_missing,
        grad_key,
        hess_key,
        thread_pool,
    ):
        local_histograms = FeatureHistogram.calculate_histogram(
            data_frame_list=cur_data_frame,
            bin_split_points=bin_split_points,
            valid_features=valid_features,
            use_missing=use_missing,
            grad_key=grad_key,
            hess_key=hess_key,
            thread_pool=thread_pool,
        )
        local_hist_bags = []
        for idx, node in enumerate(cur_to_split):
            local_hist_bags.append(
                HistogramBag(local_histograms[idx], node.id, node.parent_nodeid)
            )
        return local_hist_bags

    def cal_split_info_list(self, agg_histograms):
        if self.role == link.SERVER:
            g_histograms = []
            len_histograms = [len(x) for x in agg_histograms]
            if len(set(len_histograms)) != 1:
                raise Exception("histogram from each party must be same length")
            for node_idx in range(len(agg_histograms[0])):
                node_histograms = []
                for party_idx in range(len(agg_histograms)):
                    node_histograms.append(agg_histograms[party_idx][node_idx])
                hist_bag = reduce(lambda x, y: x + y, node_histograms)
                g_histograms.append(hist_bag)

            self.split_info_list = self.splitter.find_split(
                g_histograms, self.valid_features, self.use_missing
            )
            link.send_to_clients(
                name=self.key("split_info_list"),
                value=self.split_info_list,
                version=self._sync_version,
            )
        else:
            self.split_info_list = link.recv_from_server(
                name=self.key("split_info_list"),
                version=self._sync_version,
            )

    def fit(self):
        """Enter for homo decision tree"""
        # setup thread pool
        thread_pool = ThreadPoolExecutor(max_workers=self.num_parallel)

        logging.debug(
            'begin to fit local decision tree, tree id {}'.format(self.tree_id)
        )
        self.get_valid_features_by_tree()
        self.cal_root_node()
        tree_height = self.max_depth + 1  # non-leaf node height + 1 layer leaf

        for dep in range(tree_height):
            if self.colsample_by_level < 1:
                self.get_valid_features_by_level()

            if dep + 1 == tree_height:
                if self.role == link.CLIENT:
                    for node in self.cur_layer_node:
                        # reaching the maximum depth
                        # stop spliting and add this node to model as leaf
                        node.is_leaf = True
                        self.tree_node.append(node)
                break
            if self.role == link.CLIENT:
                logging.debug(f'start to fit layer {dep}')
                agg_local_histograms = []

                for batch_id, idx in enumerate(
                    range(0, len(self.cur_layer_node), self.max_split_nodes)
                ):
                    local_hist_bags = HomoDecisionTree.cal_local_hist_bags(
                        self.cur_layer_node[idx : idx + self.max_split_nodes],
                        self.cur_layer_datas[idx : idx + self.max_split_nodes],
                        self.bin_split_points,
                        self.valid_features,
                        self.use_missing,
                        self.grad_key,
                        self.hess_key,
                        thread_pool,
                    )
                    agg_local_histograms.extend(local_hist_bags)

                link.send_to_server(
                    name=self.key("agg_local_histograms"),
                    value=agg_local_histograms,
                    version=self._sync_version,
                )
            agg_histograms = None
            if self.role == link.SERVER:
                agg_histograms = link.recv_from_clients(
                    name=self.key("agg_local_histograms"),
                    version=self._sync_version,
                )
            self.cal_split_info_list(
                agg_histograms,
            )

            if self.role == link.CLIENT:
                new_layer_node, new_layer_data = self.update_tree(
                    self.cur_layer_node, self.split_info_list, self.cur_layer_datas
                )
                self.cur_layer_node = new_layer_node
                self.cur_layer_datas = new_layer_data

            self._sync_version += 1
        thread_pool.shutdown()
        if self.role == link.CLIENT:
            self.convert_bin_to_real()
            logging.debug(
                f'finish tree build info: iter_round={self.iter_round} group_id={self.group_id} done'
            )
            return self.tree_node
