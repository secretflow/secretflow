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
from secretflow.device.driver import reveal
from secretflow.ml.boost.sgb_v.checkpoint import SGBSnapshot, sgb_model_to_snapshot
from secretflow.ml.boost.sgb_v.model import SgbModel, from_dict
from secretflow.spec.extend.sgb_model_pb2 import (
    Common,
    EnsembleLeafWeights,
    EnsembleSplitTrees,
    Features,
)
from secretflow.spec.extend.sgb_model_pb2 import SgbModel as SgbModelPb
from secretflow.spec.extend.sgb_model_pb2 import SplitTree, TreeLeafWeights


def get_sgb_snapshot_from_pb(write_data) -> SGBSnapshot:
    sgb_model = Parse(write_data, SgbModelPb())
    sgb_model_dict = get_sgb_model_dict(sgb_model)
    model_loaded = from_dict(sgb_model_dict)

    parties = get_party_features(sgb_model)
    pyus = {p: PYU(p) for p in parties}
    df = get_sgb_model_dataframe(pyus, parties)

    snapshot = sgb_model_to_snapshot(model_loaded, df, "y")
    return snapshot


def get_sgb_model_dict(sgb_model: SgbModelPb):
    return {
        'label_holder': get_label_holder(sgb_model.label_holder),
        'common': get_common(sgb_model.common),
        'leaf_weights': get_leaf_weights(sgb_model.ensemble_leaf_weights),
        'split_trees': get_split_trees(sgb_model.ensemble_split_trees),
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


def get_party_features(sgb_model):
    parties = {}
    for party, features in sgb_model.common.partition_column.items():
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


objective_link = {
    "linear": "identity",
    'logistic': 'logit',
    'tweedie': 'log',
}


def sgb_model_to_pb(model: SgbModel, model_meta: dict) -> SgbModelPb:
    sgb_model = SgbModelPb()

    sgb_model.label_holder = model.label_holder.party
    feature_names = model_meta['feature_names']
    sgb_model.common.CopyFrom(get_common_pb(model, feature_names))

    distributed_tree_dicts = [dt.to_dict() for dt in model.trees]
    device_list = [*distributed_tree_dicts[0]['split_tree_dict'].keys()]

    sgb_model.ensemble_leaf_weights.CopyFrom(
        get_ensemble_leaf_weights_pb(distributed_tree_dicts)
    )
    set_ensemble_split_trees(sgb_model, distributed_tree_dicts, device_list)

    return sgb_model


def get_common_pb(model: SgbModel, feature_names: list[str]):
    common = Common()
    common.base = model.base
    common.link = objective_link[model.objective.value]
    common.tree_num = len(model.trees)
    partition_column_counts = model.partition_column_counts

    start_index = 0
    for party, count in partition_column_counts.items():
        features = Features()
        features.feature_names.extend(feature_names[start_index : start_index + count])
        common.partition_column[party].CopyFrom(features)
        start_index += count

    return common


def get_ensemble_leaf_weights_pb(distributed_tree_dicts):
    ensemble_leaf_weights_data = [
        tree_dict['leaf_weight'] for tree_dict in distributed_tree_dicts
    ]
    ensemble_leaf_weights_data = reveal(ensemble_leaf_weights_data)
    ensemble_leaf_weights = EnsembleLeafWeights()
    for tree_leaf_weights_pb in ensemble_leaf_weights_data:
        if isinstance(tree_leaf_weights_pb[0], list):
            tree_leaf_weights_data = [
                weight for sublist in tree_leaf_weights_pb for weight in sublist
            ]
        else:
            tree_leaf_weights_data = tree_leaf_weights_pb
        tree_leaf_weights = TreeLeafWeights()
        tree_leaf_weights.weights.extend(tree_leaf_weights_data)
        ensemble_leaf_weights.tree_leaf_weights.append(tree_leaf_weights)
    return ensemble_leaf_weights


def set_ensemble_split_trees(sgb_model, distributed_tree_dicts, device_list):
    split_trees_data = {
        device: [
            tree_dict['split_tree_dict'][device] for tree_dict in distributed_tree_dicts
        ]
        for device in device_list
    }
    for pyu, ensemble_trees in split_trees_data.items():
        ensemble_split_trees = EnsembleSplitTrees()
        ensemble_trees = reveal(ensemble_trees)
        for split_tree_data in ensemble_trees:
            split_tree = SplitTree()
            split_tree.split_features.extend(split_tree_data['split_features'])
            split_tree.split_values.extend(split_tree_data['split_values'])
            split_tree.split_indices.extend(split_tree_data['split_indices'])
            split_tree.leaf_indices.extend(split_tree_data['leaf_indices'])
            ensemble_split_trees.split_trees.append(split_tree)
        sgb_model.ensemble_split_trees[pyu.party].CopyFrom(ensemble_split_trees)
