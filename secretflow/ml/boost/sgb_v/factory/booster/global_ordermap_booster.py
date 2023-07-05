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

import copy
import logging
import time
from dataclasses import dataclass
from typing import Union

import numpy as np

from secretflow.data import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import HEU, wait

from ...model import SgbModel
from ..components import DataPreprocessor, ModelBuilder, OrderMapManager, TreeTrainer
from ..components.component import Composite, Devices, print_params
from ..sgb_actor import SGBActor


@dataclass
class GlobalOrdermapBoosterComponents:
    preprocessor: DataPreprocessor = DataPreprocessor()
    order_map_manager: OrderMapManager = OrderMapManager()
    model_builder: ModelBuilder = ModelBuilder()


@dataclass
class GlobalOrdermapBoosterParams:
    """params specifically belonged to global ordermap booster, not its components.

    num_boost_round : int, default=10
                Number of boosting iterations. Same as number of trees.
                range: [1, 1024]
    first_tree_with_label_holder_feature: bool, default=False
                Whether to train the first tree with label holder's own features.
                Can increase training speed and label security.
                The training loss may increase.
                If label holder has no feature, set this to False.
    """

    num_boost_round: int = 10
    first_tree_with_label_holder_feature: bool = False


class GlobalOrdermapBooster(Composite):
    """
    This class provides both classification and regression tree boosting (also known as GBDT, GBM)
    for vertical split dataset setting by using level wise boost.
    """

    def __init__(self, heu: HEU, tree_trainer: TreeTrainer) -> None:
        self.components = GlobalOrdermapBoosterComponents()
        self.params = GlobalOrdermapBoosterParams()
        self.user_config = {}
        self.heu = heu
        self.tree_trainer = tree_trainer

    def _set_booster_params(self, params: dict):
        trees = int(params.get('num_boost_round', 10))
        assert 1 <= trees <= 1024, f"num_boost_round should in [1, 1024], got {trees}"
        self.params.num_boost_round = trees
        self.params.first_tree_with_label_holder_feature = bool(
            params.get('first_tree_with_label_holder_feature', False)
        )

    def _get_booster_params(self, params: dict):
        params['num_boost_round'] = self.params.num_boost_round
        params[
            'first_tree_with_label_holder_feature'
        ] = self.params.first_tree_with_label_holder_feature

    def show_params(self):
        super().show_params()
        self.tree_trainer.show_params()
        print_params(self.params)

    def set_params(self, params: dict):
        super().set_params(params)
        self.tree_trainer.set_params(params)
        self._set_booster_params(params)
        self.user_config = params

    def get_params(self, params: dict = {}) -> dict:
        super().get_params(params)
        self.tree_trainer.get_params(params)
        self._get_booster_params(params)
        return params

    def set_devices(self, devices: Devices):
        super().set_devices(devices)
        self.tree_trainer.set_devices(devices)

    def set_actors(self, actors: SGBActor):
        super().set_actors(actors)
        self.tree_trainer.set_actors(actors)

    def fit(
        self,
        dataset: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
    ) -> SgbModel:
        x, x_shape, y, _ = self.components.preprocessor.validate(dataset, label)

        # set devices
        devices = Devices(y.device, [*x.partitions.keys()], self.heu)
        self.set_devices(devices)

        # set actors
        actors = [SGBActor(device=device) for device in devices.workers]
        self.set_actors(actors)

        pred = self.components.model_builder.init_pred(x_shape[0])
        self.components.order_map_manager.build_order_map(x)
        self.components.model_builder.init_model()

        for tree_index in range(self.params.num_boost_round):
            start = time.perf_counter()
            if self.params.first_tree_with_label_holder_feature and tree_index == 0:
                # we are sure the config is small, so ok to copy
                config = copy.deepcopy(self.user_config)
                config['label_holder_feature_only'] = True
                self.tree_trainer.set_params(config)
            tree = self.tree_trainer.train_tree(
                tree_index, self.components.order_map_manager, y, pred, x_shape
            )
            if tree is None:
                logging.info(
                    f"early_stopped, current tree num: {self.components.model_builder.get_tree_num()}"
                )
                break
            if self.params.first_tree_with_label_holder_feature and tree_index == 0:
                config['label_holder_feature_only'] = False
                self.tree_trainer.set_params(config)
            self.components.model_builder.insert_tree(tree)
            cur_tree_num = self.components.model_builder.get_tree_num()

            if cur_tree_num < self.params.num_boost_round:
                pred = y.device(lambda x, y: x + np.array(y, order='F'))(
                    pred, tree.predict(x.partitions)
                )
                wait([pred])
            else:
                wait(tree)

            logging.info(
                f"epoch {cur_tree_num - 1} time {time.perf_counter() - start}s"
            )

        return self.components.model_builder.finish()
