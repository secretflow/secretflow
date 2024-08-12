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
from typing import Callable, Union

from heu import phe

from secretflow.data import FedNdarray
from secretflow.data.split import train_test_split
from secretflow.data.vertical import VDataFrame
from secretflow.device import HEU
from secretflow.ml.boost.core.callback import (
    Checkpointing,
    EarlyStopping,
    EvaluationMonitor,
)
from secretflow.ml.boost.core.metric import MetricProducer
from secretflow.ml.boost.sgb_v.checkpoint import SGBCheckpointData
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
    eval_metric: str = 'roc_auc'
    enable_monitor: bool = False
    enable_early_stop: bool = False
    validation_fraction: float = 0.1
    stopping_rounds: int = 1
    stopping_tolerance: float = 0.001
    seed: int = 1212
    save_best_model: bool = False
    # only effective if objective and eval metric are related to tweedie
    tweedie_variance_power: float = 1.5


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
        self.factory_params.eval_metric = params.get('eval_metric', 'roc_auc')
        self.factory_params.enable_monitor = params.get('enable_monitor', False)
        self.factory_params.enable_early_stop = params.get('enable_early_stop', False)
        self.factory_params.validation_fraction = params.get('validation_fraction', 0.1)
        self.factory_params.stopping_rounds = params.get('stopping_rounds', 1)
        self.factory_params.stopping_tolerance = params.get('stopping_tolerance', 0.001)
        self.factory_params.seed = params.get('seed', 1212)
        self.factory_params.save_best_model = params.get('save_best_model', False)
        self.factory_params.tweedie_variance_power = params.get(
            'tweedie_variance_power', 1.5
        )

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
        assert (
            0 < self.factory_params.validation_fraction < 1
        ), f"validation fraction msut be in (0,1), got {self.factory_params.validation_fraction}"

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
        data_name: str = None,
        checkpoint_data: SGBCheckpointData = None,
        dump_function: Callable = None,
        sample_weight: Union[FedNdarray, VDataFrame] = None,
    ) -> SgbModel:
        booster = self._produce()
        callbacks = []
        eval_set = []
        metric_, metric_final_name = MetricProducer(
            self.factory_params.eval_metric,
            tweedie_variance_power=self.factory_params.tweedie_variance_power,
        )
        if self.factory_params.enable_monitor:
            callbacks.append(EvaluationMonitor())
            eval_set = [
                (dataset, label, "whole"),
            ]
        if self.factory_params.enable_early_stop:
            train_data, val_data = train_test_split(
                dataset,
                test_size=self.factory_params.validation_fraction,
                random_state=self.factory_params.seed,
            )
            train_label, val_label = train_test_split(
                label,
                test_size=self.factory_params.validation_fraction,
                random_state=self.factory_params.seed,
            )
            if sample_weight is not None:
                # weight is not used in evaluation yet, just affects training.
                train_weight, _ = train_test_split(
                    sample_weight,
                    test_size=self.factory_params.validation_fraction,
                    random_state=self.factory_params.seed,
                )

            assert val_label is not None
            callbacks.append(
                EarlyStopping(
                    self.factory_params.stopping_rounds,
                    metric_final_name,
                    data_name=data_name,
                    save_best=self.factory_params.save_best_model,
                    min_delta=self.factory_params.stopping_tolerance,
                )
            )
            eval_set = [
                (train_data, train_label, "train"),
                (val_data, val_label, "val"),
            ]
            # train using splitted data only
            dataset = train_data
            label = train_label
            sample_weight = train_weight if sample_weight is not None else None

        callbacks.append(Checkpointing(dump_function=dump_function))
        return booster.fit(
            dataset,
            label,
            callbacks=callbacks,
            eval_sets=eval_set,
            metric=metric_,
            checkpoint_data=checkpoint_data,
            sample_weight=sample_weight,
        )

    def train(
        self,
        params: dict,
        dtrain: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
        checkpoint_data: SGBCheckpointData = None,
        dump_function: Callable = None,
        sample_weight: Union[FedNdarray, VDataFrame] = None,
    ) -> SgbModel:
        """Train the SGB model

        Args:
            params (dict): sgb parameters
            dtrain (Union[FedNdarray, VDataFrame]): dataset excludes the label, must be aligned vertically
            label (Union[FedNdarray, VDataFrame]): label data, must be aligned vertically with the dtrain
            checkpoint_data (SGBCheckpointData, optional): checkpoint data used for continued training. Defaults to None.
            dump_function (Callable, optional): the dump function must accept 3 args:
                    model: CallBackCompatibleModel,
                    epoch: int,
                    evals_log: TrainingCallback.EvalsLog
                and returns nothing. It should write the model and meta info into some path specified by the user.
                This feature is now automatically supported at sf component level.
                If you don't want to use checkpoints, just leave this argument as None.
                Defaults to None.
            sample_weight (Union[FedNdarray, VDataFrame], optional): weight for each sample.
                Defaults to None. Must contain exactly one column, which belongs to label holder.
        Returns:
            SgbModel: trained SgbModel
        """
        self.set_params(params)
        # TODO: unify data type before entering this algorithm
        return self.fit(
            dtrain,
            label,
            checkpoint_data=checkpoint_data,
            dump_function=dump_function,
            sample_weight=sample_weight,
        )
