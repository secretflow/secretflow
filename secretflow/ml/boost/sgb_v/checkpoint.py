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

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.device.pyu import PYUObject
from secretflow.ml.boost.sgb_v.model import from_dict, SgbModel


@dataclass
class SGBSnapshot:
    """
    A snapshot of sgb model. Consists of model objs and meta str.
    """

    model_objs: List[PYUObject]
    model_meta: Dict


@dataclass
class SGBCheckpointData:
    """
    Essential checkpoint data of sgb model.
    In addition to all info in SGBsnapshot, it also contains train state metas information.
    """

    model_objs: List[PYUObject]
    model_train_state_metas: Dict


def build_sgb_model(snapshot: SGBSnapshot) -> SgbModel:
    """Build SgbModel from snapshot and pyus

    Args:
        snapshot (SGBSnapshot): SGBSnapshot data, contain all data sufficient to get back a SgbModel

    Returns:
        SgbModel: boosted model, ready to predict
    """

    model_objs, model_meta = (
        snapshot.model_objs,
        snapshot.model_meta,
    )
    pyus = list(set([obj.device for obj in model_objs]))
    pyus = {p.party: p for p in pyus}
    assert (
        isinstance(model_meta, dict)
        and "common" in model_meta
        and "label_holder" in model_meta
        and "tree_num" in model_meta["common"]
        and model_meta["label_holder"] in pyus
    ), f"{model_meta}, {pyus}"
    logging.info(f"model_meta check success")
    tree_num = model_meta["common"]["tree_num"]
    assert (
        tree_num > 0 and len(model_objs) % tree_num == 0
    ), f"model_objs {model_objs}, model_meta {model_meta}"
    logging.info("tree num check success")
    leaf_weights = model_objs[:tree_num]
    split_trees = {}
    for pos in range(1, int(len(model_objs) / tree_num)):
        splits = model_objs[tree_num * pos : tree_num * (pos + 1)]
        assert splits[0].device not in split_trees
        split_trees[splits[0].device] = splits
    model_meta["leaf_weights"] = leaf_weights
    model_meta["split_trees"] = split_trees
    model_meta["label_holder"] = pyus[model_meta["label_holder"]]

    model = from_dict(model_meta)
    return model


def sgb_model_to_snapshot(
    model: SgbModel, x: VDataFrame, label_name: str
) -> SGBSnapshot:
    """Convert SgbModel to SGBSnapshot"""
    m_dict = model.to_dict()
    leaf_weights = m_dict.pop("leaf_weights")
    split_trees = m_dict.pop("split_trees")
    m_dict["label_holder"] = m_dict["label_holder"].party
    m_dict["feature_names"] = x.columns
    m_dict["label_col"] = label_name
    party_features_length = {
        device.party: len(columns) for device, columns in x.partition_columns.items()
    }
    m_dict["party_features_length"] = party_features_length

    m_objs = sum([leaf_weights, *split_trees.values()], [])
    return SGBSnapshot(m_objs, m_dict)


def snapshot_to_checkpoint_data(
    snapshot: SGBSnapshot, train_state: Dict
) -> SGBCheckpointData:
    combined = {
        'model_meta': snapshot.model_meta,
        'train_state': train_state,
    }

    return SGBCheckpointData(snapshot.model_objs, combined)


def sgb_model_to_checkpoint_data(
    model: SgbModel, train_state: Dict, x: VDataFrame, label_name: str
) -> SGBCheckpointData:
    return snapshot_to_checkpoint_data(
        sgb_model_to_snapshot(model, x, label_name), train_state
    )


def checkpoint_data_to_model_and_train_state(
    checkpoint_data: SGBCheckpointData,
) -> Tuple[SgbModel, Dict]:
    combined_dict = checkpoint_data.model_train_state_metas
    model_meta = combined_dict['model_meta']
    train_state = combined_dict['train_state']
    return (
        build_sgb_model(SGBSnapshot(checkpoint_data.model_objs, model_meta)),
        train_state,
    )
