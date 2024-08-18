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
from abc import ABC, abstractmethod
from typing import Dict, List

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.attacks.base import AttackBase
from benchmark_examples.autoattack.base import AutoBase
from benchmark_examples.autoattack.utils.config import read_tune_config
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow import PYU
from secretflow.ml.nn.callbacks.callback import Callback


class DefenseBase(AutoBase, ABC):
    config: Dict
    alice: PYU
    bob: PYU

    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def set_config(self, config: Dict[str, str] | None):
        super().set_config(config)

    @abstractmethod
    def build_defense_callback(self, app: ApplicationBase) -> Callback | None:
        pass

    def search_space(self) -> Dict[str, List]:
        tune_config: dict = read_tune_config(global_config.get_config_file_path())
        assert (
            'defenses' in tune_config
        ), f"Missing 'defenses' after 'tune' in config file."
        defense_config = tune_config['defenses']
        assert (
            self.__str__() in defense_config
        ), f"Missing tune.defenses.{self.__str__()} in config file."
        defense_search_space = defense_config[self.__str__()]
        defense_search_space = (
            {} if defense_search_space is None else defense_search_space
        )
        if global_config.is_simple_test():
            # delete some defense config for speed up.
            new_defense_search_space = {}
            nums = 0
            for k, v in defense_search_space.items():
                if nums > 2:
                    break
                if isinstance(v, list) and len(v) > 2:
                    new_defense_search_space[k] = v[0:2]
                else:
                    new_defense_search_space[k] = v
                nums += 1
            defense_search_space = new_defense_search_space
        return defense_search_space

    def check_attack_valid(self, attack: AttackBase) -> bool:
        return False

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return False

    def update_resources_consumptions(
        self,
        cluster_resources_pack: ResourcesPack,
        app: ApplicationBase,
        attack: AttackBase | None,
    ) -> ResourcesPack:
        """Update the resource consumptions depends on each defense."""
        pass

    def tune_metrics(self, app_metrics: Dict[str, str]) -> Dict[str, str]:
        """
        Return the defense tune metrics, and can modify the app metircs.
        Args:
            app_metrics: application metrics to modify.

        Returns:
            the defense tuen metrics.
        """


class DefaultDefenseCase(DefenseBase):
    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def tune_metrics(self, app_metrics: Dict[str, str]) -> Dict[str, str]:
        return {}

    def __str__(self):
        return ""

    def build_defense_callback(
        self, app: ApplicationBase, attack: AttackBase | None = None
    ) -> Callback | None:
        return None
