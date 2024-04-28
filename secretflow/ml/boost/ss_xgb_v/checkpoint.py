# Copyright 2024 Ant Group Co., Ltd.
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

from dataclasses import dataclass
from typing import Dict, List

from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import SPU
from secretflow.ml.boost.ss_xgb_v.core.node_split import RegType
from secretflow.ml.boost.ss_xgb_v.model import XgbModel


@dataclass
class SSXGBCheckpointData:
    """
    Essential checkpoint data of ss xgb model.
    """

    model_objs: List
    model_metas: Dict


def ss_xgb_model_to_checkpoint_data(
    model: XgbModel, x: VDataFrame, train_dataset_label: str
) -> SSXGBCheckpointData:
    """Convert XgbModel to SSXGBCheckpointData."""
    m_dict = {
        "objective": model.objective.value,
        "base": model.base,
        "tree_num": len(model.weights),
        "feature_names": x.columns,
        "label_col": train_dataset_label,
    }
    party_features_length = {
        device.party: len(columns) for device, columns in x.partition_columns.items()
    }
    m_dict["party_features_length"] = party_features_length

    split_trees = []
    for p in x.partitions.keys():
        split_trees.extend([t[p] for t in model.trees])
    return SSXGBCheckpointData([*model.weights, *split_trees], m_dict)


def build_ss_xgb_model(checkpoint_data: SSXGBCheckpointData, spu: SPU) -> XgbModel:
    model_objs = checkpoint_data.model_objs
    model_meta = checkpoint_data.model_metas
    assert (
        isinstance(model_meta, dict)
        and "objective" in model_meta
        and "base" in model_meta
        and "tree_num" in model_meta
    )
    tree_num = model_meta["tree_num"]
    assert (
        tree_num > 0 and len(model_objs) % tree_num == 0
    ), f"model_objs {model_objs}, model_meta {model_meta}"
    weights = model_objs[:tree_num]
    trees = []
    parties_num = int(len(model_objs) / tree_num) - 1
    for pos in range(tree_num):
        tree = {}
        for p in range(parties_num):
            obj = model_objs[tree_num * (p + 1) + pos]
            tree[obj.device] = obj
        trees.append(tree)

    model = XgbModel(spu, RegType(model_meta["objective"]), model_meta["base"])
    model.weights = weights
    model.trees = trees

    return model
