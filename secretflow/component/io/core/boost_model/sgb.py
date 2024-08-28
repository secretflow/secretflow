# Copyright 2024 Ant Group Co., Ltd.
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
from typing import List

import pandas as pd
from google.protobuf.json_format import Parse

from secretflow.data.core import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.device.pyu import PYU
from secretflow.ml.boost.sgb_v.checkpoint import SGBSnapshot, sgb_model_to_snapshot
from secretflow.ml.boost.sgb_v.model import from_dict
from secretflow.spec.extend.xgb_model_pb2 import (
    Common,
    EnsembleLeafWeights,
    EnsembleSplitTrees,
    XgbModel,
)


def get_sgb_snapshot_from_pb(write_data) -> SGBSnapshot:
    xgb_model = Parse(write_data, XgbModel())
    sgb_model_dict = get_sgb_model_dict(xgb_model)
    model_loaded = from_dict(sgb_model_dict)

    parties = get_party_features(xgb_model)
    pyus = {p: PYU(p) for p in parties}
    df = get_sgb_model_dataframe(pyus, parties)

    snapshot = sgb_model_to_snapshot(model_loaded, df, "y")
    return snapshot


def get_sgb_model_dict(xgb_model: XgbModel):
    return {
        'label_holder': get_label_holder(xgb_model.label_holder),
        'common': get_common(xgb_model.common),
        'leaf_weights': get_leaf_weights(xgb_model.ensemble_leaf_weights),
        'split_trees': get_split_trees(xgb_model.ensemble_split_trees),
    }


def get_label_holder(label_holder: str):
    return PYU(label_holder)


def get_common(common: Common):
    parties_count = {}
    for party, features in common.partition_column.items():
        parties_count[party] = len(features.feature_names)

    common_params = {}
    common_params['base'] = common.base
    common_params['objective'] = convert_link_2_objective(common.link)
    common_params['partition_column_counts'] = parties_count
    common_params['tree_num'] = common.tree_num
    return common_params


def get_leaf_weights(ensemble_leaf_weights_pb: EnsembleLeafWeights):
    ensemble_leaf_weights = []
    for _, tree_weights in enumerate(ensemble_leaf_weights_pb.tree_leaf_weights):
        leaf_weights = []
        for _, leaf_weight in enumerate(tree_weights.weights):
            leaf_weights.append(leaf_weight)
        ensemble_leaf_weights.append(leaf_weights)
    return ensemble_leaf_weights


def get_split_trees(all_party_ensemble_split_trees_pb: EnsembleSplitTrees):
    all_party_ensemble_split_trees = {}
    for party, ensemble_split_trees_pb in all_party_ensemble_split_trees_pb.items():
        logging.warning(f"ensemble_split_trees Party: {party}")
        ensemble_split_trees = []
        for _, split_tree_pb in enumerate(ensemble_split_trees_pb.split_trees):
            split_tree = {
                'split_features': list(split_tree_pb.split_features),
                'split_values': list(split_tree_pb.split_values),
                'split_indices': list(split_tree_pb.split_indices),
                'leaf_indices': list(split_tree_pb.leaf_indices),
            }
            # logging.warning(f"  Split tree: {split_tree}")
            ensemble_split_trees.append(split_tree)
        all_party_ensemble_split_trees[PYU(party)] = ensemble_split_trees
    return all_party_ensemble_split_trees


def get_party_features(xgb_model):
    parties = {}
    for party, features in xgb_model.common.partition_column.items():
        parties[party] = features.feature_names
    return parties


def get_sgb_model_dataframe(pyus: dict[str, PYU], party_features: dict[str, List[str]]):
    partitions = {}
    for party, pyu in pyus.items():
        df = pd.DataFrame(
            data=[],
            columns=party_features[party],
        )
        partitions[pyu] = partition(pyu(lambda x: x)(df))

    df = VDataFrame(partitions=partitions)
    return df


def convert_link_2_objective(link: str):
    if link == "identity":
        return "linear"
    elif link == "logit":
        return "logistic"
    elif link == "log":
        return "tweedie"
    else:
        raise ValueError(f"Unsupported link: {link}")
