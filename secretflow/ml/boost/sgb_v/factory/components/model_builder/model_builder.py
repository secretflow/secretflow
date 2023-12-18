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

from secretflow.device import PYUObject
from secretflow.ml.boost.sgb_v.core.params import default_params

from ....core.distributed_tree.distributed_tree import DistributedTree
from ....core.params import RegType
from ....core.pure_numpy_ops.pred import init_pred
from ....model import SgbModel
from ..component import Component, Devices, print_params


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
        obj = params.get('objective', default_params.objective.value)
        obj = RegType(obj)
        self.params.objective = obj
        self.params.base_score = params.get('base_score', default_params.base_score)

    def get_params(self, params: dict):
        params['base_score'] = self.params.base_score
        params['objective'] = self.params.objective

    def set_devices(self, devices: Devices):
        self.label_holder = devices.label_holder

    def set_actors(self, _):
        return

    def del_actors(self):
        return

    def init_pred(self, sample_num: Union[PYUObject, int]) -> PYUObject:
        base = self.params.base_score
        return self.label_holder(init_pred)(base=base, samples=sample_num)

    def init_model(self):
        self.model = SgbModel(
            self.label_holder, self.params.objective, self.params.base_score
        )

    def insert_tree(self, tree: DistributedTree):
        self.model._insert_distributed_tree(tree)

    def get_tree_num(self) -> int:
        return len(self.model.trees)

    def finish(self) -> SgbModel:
        return self.model
