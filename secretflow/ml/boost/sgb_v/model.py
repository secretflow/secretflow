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
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
import numpy as np

from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, PYUObject, reveal, wait
from secretflow.ml.boost.core.data_preprocess import prepare_dataset

from .core.distributed_tree.distributed_tree import DistributedTree
from .core.distributed_tree.distributed_tree import from_dict as dt_from_dict
from .core.params import RegType
from .core.pure_numpy_ops.pred import sigmoid

common_path_postfix = "/common.json"
leaf_weight_postfix = "/leaf_weight.json"
split_tree_postfix = "/split_tree.json"


class SgbModel:
    """
    Sgboost Model & predict. It is a distributed tree in essence.
    """

    def __init__(self, label_holder: PYU, objective: RegType, base: float) -> None:
        """
        Args:
            label_holder: PYU device, label holder's PYU device.
            objective: RegType, specifies doing logistic regression or regression
            base: float
        """
        self.label_holder = label_holder
        self.objective = objective
        self.base = base
        self.trees: List[DistributedTree] = list()
        # useful for dumping to a complete model or check data shape before predict.
        # TODO(zoupeicheng.zpc): check input data columns count match
        self.partition_column_counts = {}

    def _insert_distributed_tree(self, tree: DistributedTree):
        self.trees.append(tree)

    def __getitem__(self, index) -> 'SgbModel':
        model = SgbModel(self.label_holder, self.objective, self.base)
        model.trees = self.trees[index]
        model.partition_column_counts = self.partition_column_counts
        return model

    def predict_with_trees(self, dtrain: Union[FedNdarray, VDataFrame]) -> PYUObject:
        if len(self.trees) == 0:
            return None

        pred = 0
        x, _ = prepare_dataset(dtrain)
        x = x.partitions

        for tree in self.trees:
            pred = self.label_holder(lambda x, y: jnp.add(x, y))(tree.predict(x), pred)

        pred = self.label_holder(lambda x, y: jnp.add(x, y).reshape(-1, 1))(
            pred, self.base
        )
        return pred

    def apply_activation(self, pred: PYUObject) -> PYUObject:
        if self.objective == RegType.Logistic:
            pred = self.label_holder(sigmoid)(pred)
        elif self.objective == RegType.Tweedie:
            pred = self.label_holder(lambda x: np.exp(x))(pred)
        return pred

    def predict(
        self,
        dtrain: Union[FedNdarray, VDataFrame],
        to_pyu: PYU = None,
    ) -> Union[PYUObject, FedNdarray]:
        """
        predict on dtrain with this model.

        Args:

            dtrain : [FedNdarray, VDataFrame]
                vertical split dataset.

            to: the prediction initiator
                if not None predict result is reveal to to_pyu device and save as FedNdarray
                otherwise, keep predict result in plaintext and save as PYUObject in label_holder device.

        Return:
            Pred values store in pyu object or FedNdarray. PYUObject by default, FedNdarray if to_pyu is not None.
        """
        pred = self.predict_with_trees(dtrain)
        pred = self.apply_activation(pred)
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

    def to_dict(self) -> Dict:
        distributed_tree_dicts = [dt.to_dict() for dt in self.trees]
        device_list = [*distributed_tree_dicts[0]['split_tree_dict'].keys()]

        return {
            'label_holder': self.label_holder,
            'common': {
                'objective': self.objective.value,
                'base': self.base,
                'tree_num': len(self.trees),
                'partition_column_counts': self.partition_column_counts,
            },
            'leaf_weights': [
                tree_dict['leaf_weight'] for tree_dict in distributed_tree_dicts
            ],
            'split_trees': {
                device: [
                    tree_dict['split_tree_dict'][device]
                    for tree_dict in distributed_tree_dicts
                ]
                for device in device_list
            },
        }

    def sync_partition_columns_to_all_distributed_trees(self):
        for tree in self.trees:
            tree.partition_column_counts = self.partition_column_counts

    def save_model(self, device_path_dict: Dict, wait_before_proceed=True):
        """Save model to different parties

        Args:
            device_path_dict (Dict): {device: a path to save model for the device}.
            wait_before_process (bool): if False, handle will be returned,
                to allow user to wait for model write to finish
                (and do something else in the meantime).
        """

        def json_dump(obj, path):
            f = Path(path)
            f.parent.mkdir(exist_ok=True, parents=True)
            with open(path, 'w') as f:
                json.dump(obj, f)
            return None

        model_dict = self.to_dict()
        assert (
            self.label_holder in device_path_dict
        ), "device holder path must be provided"

        # save common
        common_path = device_path_dict[self.label_holder] + common_path_postfix
        finish_common = self.label_holder(json_dump)(model_dict['common'], common_path)

        # save leaf weight
        leaf_weight_path = device_path_dict[self.label_holder] + leaf_weight_postfix
        finish_leaf = self.label_holder(json_dump)(
            model_dict['leaf_weights'], leaf_weight_path
        )

        # save split trees
        finish_split_trees = []
        for device, path in device_path_dict.items():
            split_tree_path = path + split_tree_postfix
            if device in model_dict['split_trees']:
                finish_split_trees.append(
                    device(json_dump)(
                        model_dict['split_trees'][device], split_tree_path
                    )
                )

        # no real content, handler for wait
        r = (finish_common, finish_leaf, finish_split_trees)
        if wait_before_proceed:
            wait(r)
            return None
        return r

    def get_trees(self):
        return self.trees

    def get_objective(self):
        return self.objective

    def feature_importance_device_wise(self) -> Dict[PYU, Dict[int, float]]:
        """Get feature importance of each feature.

        Returns:
            Dict: {device: {feature_index: importance}}
        """
        if len(self.trees) == 0:
            logging.warning("No trees found, cannot get feature importance.")
            return None
        gain_stats_list = [tree.gain_statistics() for tree in self.trees]
        device_list = gain_stats_list[0].keys()
        feature_importance_device_wise = {}
        for device in device_list:
            gain_stats_device = [
                gain_stats_list[i][device] for i in range(len(gain_stats_list))
            ]
            agg_gain_statistics_device = reveal(
                device(agg_gain_statistics)(gain_stats_device)
            )
            feature_importance_device_wise[device] = agg_gain_statistics_device
        return feature_importance_device_wise

    def feature_importance_flatten(
        self, x: Union[VDataFrame, FedNdarray]
    ) -> np.ndarray:
        if not isinstance(x, VDataFrame) and not isinstance(x, FedNdarray):
            raise ValueError(
                "x must be a VDataFrame or FedNdarry, please input the training data set."
            )
        feature_importance_device_wise = self.feature_importance_device_wise()
        feature_importance_flatten = np.zeros(x.shape[1])
        offset = 0
        for device, shape in x.partition_shape().items():
            for i in range(shape[1]):
                feature_importance_flatten[offset + i] = feature_importance_device_wise[
                    device
                ][i]
            offset += shape[1]
        return feature_importance_flatten


def agg_gain_statistics(gain_stats_list: List[Tuple[Dict, Dict]]):
    """Aggregate gain statistics from different trees"""
    gain_sums = {}
    gain_count = {}
    for tree_gain_sum, tree_gain_count in gain_stats_list:
        for feature_name, gain in tree_gain_sum.items():
            if feature_name == -1:
                continue
            if feature_name not in gain_sums:
                gain_sums[feature_name] = 0
                gain_count[feature_name] = 0
            gain_sums[feature_name] += gain
            gain_count[feature_name] += tree_gain_count[feature_name]

    return {k: v / max([gain_count[k], 1]) for k, v in gain_sums.items()}


def from_dict(model_dict: Dict) -> SgbModel:
    sm = SgbModel(
        model_dict['label_holder'],
        RegType(model_dict['common']['objective']),
        model_dict['common']['base'],
    )
    sm.partition_column_counts = model_dict['common']['partition_column_counts']
    device_list = [*model_dict['split_trees'].keys()]

    def build_split_tree_dict(i):
        return {device: model_dict['split_trees'][device][i] for device in device_list}

    sm.trees = [
        dt_from_dict(
            {
                'split_tree_dict': build_split_tree_dict(i),
                'leaf_weight': leaf_weight,
                'label_holder': model_dict['label_holder'],
                'partition_column_counts': model_dict['common'][
                    'partition_column_counts'
                ],
            }
        )
        for i, leaf_weight in enumerate(model_dict['leaf_weights'])
    ]
    return sm


def check_file_exists(path):
    return True if os.path.isfile(path) else False


def from_json_to_dict(
    device_path_dict: Dict,
    label_holder: PYU,
) -> Dict:
    def json_load(path):
        with open(path, 'r') as f:
            r = json.load(f)
        return r

    assert label_holder in device_path_dict, "device holder path must be provided"

    # load common
    common_path = device_path_dict[label_holder] + common_path_postfix
    common_params = reveal(label_holder(json_load)(common_path))

    # load leaf weight
    leaf_weight_path = device_path_dict[label_holder] + leaf_weight_postfix
    if common_params['tree_num'] > 1:
        leaf_weights = [
            *label_holder(json_load, num_returns=common_params['tree_num'])(
                leaf_weight_path
            )
        ]
    else:
        leaf_weights = [label_holder(lambda x: json_load(x)[0])(leaf_weight_path)]
    # check split trees
    split_devices = {}
    for device, path in device_path_dict.items():
        if reveal(device(check_file_exists)(path + split_tree_postfix)):
            split_devices[device] = path

    if common_params['tree_num'] > 1:
        split_tree_dict = {
            device: [
                *device(json_load, num_returns=common_params['tree_num'])(
                    path + split_tree_postfix
                )
            ]
            for device, path in split_devices.items()
        }

    else:
        split_tree_dict = {
            device: [device(lambda x: json_load(x)[0])(path + split_tree_postfix)]
            for device, path in split_devices.items()
        }

    return {
        'label_holder': label_holder,
        'common': common_params,
        'leaf_weights': leaf_weights,
        'split_trees': split_tree_dict,
    }


def load_model(
    device_path_dict: Dict,
    label_holder: PYU,
) -> SgbModel:
    return from_dict(from_json_to_dict(device_path_dict, label_holder))
