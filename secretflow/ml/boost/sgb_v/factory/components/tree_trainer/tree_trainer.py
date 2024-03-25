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

import abc
from typing import List

from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

from ....core.distributed_tree.distributed_tree import DistributedTree
from ..component import Composite, Devices


class TreeTrainer(Composite):
    def show_params(self):
        super().show_params()

    def set_params(self, params: dict):
        super().set_params(params)

    def set_devices(self, devices: Devices):
        super().set_devices(devices)

    def set_actors(self, actors: List[SGBActor]):
        return super().set_actors(actors)

    @abc.abstractmethod
    def train_tree(
        self, cur_tree_num, order_map_manager, y, pred, sample_num
    ) -> DistributedTree:
        """train on training data"""
        pass
