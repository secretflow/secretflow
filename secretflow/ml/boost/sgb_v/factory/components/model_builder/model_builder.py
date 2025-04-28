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

from dataclasses import dataclass
from typing import Union

from secretflow.data import FedNdarray
from secretflow.device import PYUObject
from secretflow.ml.boost.sgb_v.core.params import default_params

from ....core.distributed_tree.distributed_tree import DistributedTree
from ....core.params import RegType
from ....core.pure_numpy_ops.pred import init_pred
from ....model import SgbModel
from ..component import Component, Devices, print_params, set_dict_from_params


@dataclass
class ModelBuilderParams:
    """
    'objective': Specify the learning objective.
        default: 'logistic'
        range: ['linear', 'logistic']
    'base_score': The initial prediction score of all instances, global bias.
        default: 0
    """

    base_score: float = default_params.base_score
    objective: RegType = default_params.objective


class ModelBuilder(Component):
    """Functions related to build models including making predictions and add trees"""

    def __init__(self):
        self.params = ModelBuilderParams()

    def show_params(self):
        print_params(self.params)

    def set_params(self, params: dict):
        if 'objective' in params:
            self.params.objective = RegType(params['objective'])
        if 'base_score' in params:
            self.params.base_score = params['base_score']

    def get_params(self, params: dict):
        set_dict_from_params(self.params, params)

    def set_devices(self, devices: Devices):
        self.label_holder = devices.label_holder

    def set_actors(self, _):
        return

    def del_actors(self):
        return

    def init_pred(
        self,
        sample_num: Union[PYUObject, int],
        checkpoint_model: SgbModel = None,
        x: FedNdarray = None,
    ) -> PYUObject:
        if checkpoint_model is None:
            base = self.params.base_score
            return self.label_holder(init_pred)(base=base, samples=sample_num)
        else:
            assert x is not None, "x must be provided"
            return checkpoint_model.predict_with_trees(x)

    def init_model(self, checkpoint_model: SgbModel = None):
        self.model = (
            SgbModel(self.label_holder, self.params.objective, self.params.base_score)
            if checkpoint_model is None
            else checkpoint_model
        )

    def insert_tree(self, tree: DistributedTree):
        self.model._insert_distributed_tree(tree)

    def set_parition_shapes(self, x: FedNdarray):
        shapes = x.partition_shape()
        self.model.partition_column_counts = {
            device.party: shape[1] for device, shape in shapes.items()
        }

    def get_tree_num(self) -> int:
        return len(self.model.trees)

    def finish(self) -> SgbModel:
        self.model.sync_partition_columns_to_all_distributed_trees()
        return self.model
