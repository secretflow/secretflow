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
from enum import Enum
from typing import Union

from secretflow.data import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import HEU

from ..model import SgbModel
from .booster import GlobalOrdermapBooster
from .components import LevelWiseTreeTrainer


class TreeGrowingMethod(Enum):
    LEVEL = "level"
    LEAF = "leaf"


@dataclass
class SGBFactoryParams:
    tree_growing_method: TreeGrowingMethod = TreeGrowingMethod.LEVEL


class SGBFactory:
    """You can customize your own boosting algorithms which are based on any combination of ideas of secureboost, XGB, and lightGBM.
    The parameters for the produced booster algorithm depends on what components it consists of. See components' parameters.

    Attributes:
        params_dict (dict): A dict contain params for the factory, booster and its components.
        factory_params (SGBFactoryParams): validated params for the factory.
        heu: the device for HE computations. must be set before training.
    """

    def __init__(self):
        self.params_dict = {'tree_growing_method': "level"}
        self.factory_params = SGBFactoryParams()
        self.heu = None

    def set_params(self, params: dict):
        """Set params by a dictionary."""
        tree_grow_method = params.get('tree_growing_method', "level")
        assert (
            tree_grow_method == "level" or tree_grow_method == "leaf"
        ), f"tree growing method must one one of 'level' or 'leaf', got {tree_grow_method}"
        self.params_dict = params
        self.factory_params.tree_growing_method = TreeGrowingMethod(tree_grow_method)

    def set_heu(self, heu: HEU):
        self.heu = heu

    def _produce(self) -> GlobalOrdermapBooster:
        assert self.heu is not None, "HEU must be set"
        if self.factory_params.tree_growing_method == TreeGrowingMethod.LEVEL:
            tree_trainer = LevelWiseTreeTrainer()
            booster = GlobalOrdermapBooster(self.heu, tree_trainer)
            # this line rectifies any conflicts in default settting of components
            booster.set_params(booster.get_params())
            # apply any custom settings
            booster.set_params(self.params_dict)
            return booster
        else:
            assert False, "Feature not supported yet"

    def get_params(self, detailed: bool = False) -> dict:
        """get the params set

        Args:
            detailed (bool, optional): If include default settings. Defaults to False.

        Returns:
            dict: current params.
        """
        if detailed:
            # detailed option will include all defaults
            if self.factory_params.tree_growing_method == TreeGrowingMethod.LEVEL:
                tree_trainer = LevelWiseTreeTrainer()
                booster = GlobalOrdermapBooster(self.heu, tree_trainer)
                # this line rectifies any conflicts in default settting of components
                booster.set_params(booster.get_params())
                # apply any custom settings
                booster.set_params(self.params_dict)
                return booster.get_params()
            else:
                assert False, "Feature not supported yet"
        else:
            # show only customized params
            return self.params_dict

    def fit(
        self,
        dataset: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
    ) -> SgbModel:
        booster = self._produce()
        return booster.fit(dataset, label)

    def train(
        self,
        params: dict,
        dataset: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
    ) -> SgbModel:
        self.set_params(params)
        return self.fit(dataset, label)
