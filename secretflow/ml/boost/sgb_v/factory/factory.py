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

import logging
from dataclasses import dataclass
from typing import Union

from heu import phe

from secretflow.data import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import HEU
from secretflow.ml.boost.sgb_v.core.params import (
    TreeGrowingMethod,
    default_params,
    get_unused_params,
    type_and_range_check,
)
from secretflow.ml.boost.sgb_v.factory.components.logging import logging_params_names

from ..model import SgbModel
from .booster import GlobalOrdermapBooster
from .components import LeafWiseTreeTrainer, LevelWiseTreeTrainer


@dataclass
class SGBFactoryParams:
    tree_growing_method: TreeGrowingMethod = default_params.tree_growing_method


class SGBFactory:
    """You can customize your own boosting algorithms which are based on any combination of ideas of secureboost, XGB, and lightGBM.
    The parameters for the produced booster algorithm depends on what components it consists of. See components' parameters.

    Attributes:
        params_dict (dict): A dict contain params for the factory, booster and its components.
        factory_params (SGBFactoryParams): validated params for the factory.
        heu: the device for HE computations. must be set before training.
    """

    def __init__(self, heu=None):
        # params_dict is either default or user set, should not change by program
        self.params_dict = {'tree_growing_method': default_params.tree_growing_method}
        self.factory_params = SGBFactoryParams()
        self.heu = heu

    def set_params(self, params: dict):
        """Set params by a dictionary."""
        type_and_range_check(params)
        unused_params = get_unused_params(params)
        unused_params -= logging_params_names
        if len(unused_params) > 0:
            logging.warning(f"The following params are not effective: {unused_params}")

        tree_grow_method = params.get(
            'tree_growing_method', default_params.tree_growing_method
        )
        self.params_dict = params
        self.factory_params.tree_growing_method = TreeGrowingMethod(tree_grow_method)

    def set_heu(self, heu: HEU):
        self.heu = heu

    def _produce(self) -> GlobalOrdermapBooster:
        assert self.heu is not None, "HEU must be set"
        if self.heu.schema == phe.parse_schema_type("elgamal"):
            assert self.params_dict.get(
                'enable_quantization', False
            ), "When the schema is elgamal, we must enable quantization to avoid runtime errors."
            assert self.params_dict.get(
                'quantization_scale', default_params.quantization_scale
            ) * (
                1
                << self.params_dict.get(
                    'fixed_point_parameter', default_params.fixed_point_parameter
                )
            ) < (
                # this value is set in the HEU, later may become configurable
                1
                << 32
            ), "quantization scale and fix point parameter too large for elgamal scheme, try to set them lower"
        if self.factory_params.tree_growing_method == TreeGrowingMethod.LEVEL:
            tree_trainer = LevelWiseTreeTrainer()
        else:
            tree_trainer = LeafWiseTreeTrainer()
        booster = GlobalOrdermapBooster(self.heu, tree_trainer)
        # this line rectifies any conflicts in default settting of components
        booster.set_params(booster.get_params())
        # apply any custom settings
        booster.set_params(self.params_dict)
        return booster

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
            else:
                tree_trainer = LeafWiseTreeTrainer()
            booster = GlobalOrdermapBooster(self.heu, tree_trainer)
            # this line rectifies any conflicts in default settting of components
            booster.set_params(booster.get_params())
            # apply any custom settings
            booster.set_params(self.params_dict)
            return booster.get_params()
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
        dtrain: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
    ) -> SgbModel:
        self.set_params(params)
        return self.fit(dtrain, label)
