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

from abc import abstractmethod
from enum import Enum
from typing import Dict, Tuple

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.base import AutoBase
from benchmark_examples.autoattack.utils.config import read_tune_config
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.attack import AttackCallback


class AttackType(Enum):
    """The attack types."""

    LABLE_INFERENSE = 1
    FEATURE_INFERENCE = 2
    BACKDOOR = 3
    OTHER = 4


class AttackBase(AutoBase):
    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def set_config(self, config: Dict[str, str] | None):
        super().set_config(config)

    @abstractmethod
    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback | None:
        """
        Given the application configuration, construct a callback for the attack based on the configuration
        Args:
            app: The target implementation of application, witch has its own configuration of this attack.

        Returns:
            The attack callback.
        """
        pass

    @abstractmethod
    def attack_type(self) -> AttackType:
        """
        Get the type of this attack, refere to AttackType.
        Returns:
            The attack type.
        """
        pass

    def attack_metrics_params(self) -> Tuple | None:
        """
        If the attack metrics need some extra parameters (such as the preds after predict phase),
        return them in this metrics params.
        Returns:
            Any number of Tuple parameters, which will be passed to the AttackCallback.get_attack_metrics().
        """
        return None

    def search_space(self):
        tune_config: dict = read_tune_config(global_config.get_config_file_path())
        assert (
            'attacks' in tune_config
        ), f"Missing 'attacks' after 'tune' in config file."
        attack_config = tune_config['attacks']
        assert (
            self.__str__() in attack_config
        ), f"Missing {self.__str__()} in config file."
        attack_search_space = attack_config[self.__str__()]
        attack_search_space = {} if attack_search_space is None else attack_search_space
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
        return attack_search_space

    def check_app_valid(self, app: ApplicationBase) -> bool:
        """Chekck whether the attack support the application or not."""
        return False

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        """Update the resource consumptions depends on each attack."""
        pass


class DefaultAttackCase(AttackBase):

    def __str__(self):
        return ""

    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback | None:
        return None

    def tune_metrics(self) -> Dict[str, str]:
        return {}

    def attack_type(self) -> AttackType:
        return AttackType.OTHER
