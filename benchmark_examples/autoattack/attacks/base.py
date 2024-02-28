# Copyright 2024 Ant Group Co., Ltd.
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
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.utils.sync_globals import sync_remote_globals
from secretflow import PYU, tune


class AttackCase(ABC):
    """
    Abstract base class for attack case.
    Since the self.attack() will be tuned and be executed remote.
    For correctly inject configs into App, the self.app need to be inited into remote function,
    so we addtinally need a config_app instance outsides.
    args:
        config (dict): Configuration dict for tune.
        alice: (PYU): alice party.
        bob: (PYU): bob party.
        App: (cls): The application class for init.
        config_app (ApplicationBase): An instance of class ApplicationBase just for read tunning configs.
        app (ApplicationBase): The real application instant for training.
    """

    config: Dict
    alice: PYU
    bob: PYU
    App: type(ApplicationBase)
    config_app: ApplicationBase
    app: ApplicationBase
    origin_global_configs: Dict

    def __init__(
        self, alice, bob, App: type(ApplicationBase), origin_global_configs: dict = None
    ):
        self.App = App
        self.alice = alice
        self.bob = bob
        self.config_app: ApplicationBase = self.App({}, self.alice, self.bob)
        self.app: Optional[ApplicationBase] = None
        self.config: Optional[dict] = None
        self.origin_global_configs = origin_global_configs

    def attack(self, config: dict):
        if self.origin_global_configs is not None:
            sync_remote_globals(self.origin_global_configs)
        # this function will be executed remote, so app need to be initialized again.
        self.app = self.App(config, self.alice, self.bob)
        self.config = config
        histories, attack_metrics = self._attack()
        metrics = {}
        # append the origin application train history to the attack metrics, for record.
        for history, v in histories.items():
            metrics[f"app_{history}"] = v[-1]
        metrics.update(attack_metrics)
        return metrics

    @abstractmethod
    def _attack(self):
        pass

    def search_space(self):
        """
        Search space for tunning.
        Returns:
            Application search space + attack search space.
        """
        search_space = {
            'train_batch_size': [64, 128, 256],
            'alice_fea_nums': self.config_app.alice_feature_nums_range(),
            'hidden_size': self.config_app.hidden_size_range(),
            'dnn_base_units_size_alice': self.config_app.dnn_base_units_size_range_alice(),
            'dnn_base_units_size_bob': self.config_app.dnn_base_units_size_range_bob(),
            'dnn_fuse_units_size': self.config_app.dnn_fuse_units_size_range(),
            'dnn_embedding_dim': self.config_app.dnn_embedding_dim_range(),
            'deepfm_embedding_dim': self.config_app.deepfm_embedding_dim_range(),
        }
        if global_config.is_simple_test():
            search_space['train_batch_size'] = [32]
            search_space.pop('hidden_size')
            search_space.pop('dnn_base_units_size_bob')
            search_space.pop('dnn_base_units_size_alice')
            search_space.pop('dnn_fuse_units_size')
            search_space.pop('dnn_embedding_dim')
            search_space.pop('deepfm_embedding_dim')
        attack_search_space = self.attack_search_space()
        if global_config.is_simple_test():
            # delete some config for speed up.
            new_attack_search_space = {}
            nums = 0
            for k, v in attack_search_space.items():
                if nums > 2:
                    break
                if isinstance(v, list) and len(v) > 2:
                    new_attack_search_space[k] = v[0:2]
                else:
                    new_attack_search_space[k] = v
                nums += 1
            attack_search_space = new_attack_search_space
        app_search_space = {}
        app_search_space.update(attack_search_space)

        for k, v in search_space.items():
            if v is not None and isinstance(v, list):
                if len(v) > 1:
                    app_search_space[k] = tune.search.grid_search(v)
            else:
                # need any config even is None, since None will rewrite the default configs.
                app_search_space[k] = v
        logging.warning(
            f"The {type(self.config_app).__name__} {type(self).__name__} search space is {app_search_space}"
        )
        return app_search_space

    @abstractmethod
    def attack_search_space(self):
        pass

    @abstractmethod
    def metric_name(self) -> Union[str, List[str]]:
        """
        The attack metric name or names.
        Returns:
            str or list of metric names.
        """
        pass

    @abstractmethod
    def metric_mode(self) -> Union[str, List[str]]:
        """
        The attack metric mode or modes.
        Returns:
            str or list of metric modes.
        """
        pass
