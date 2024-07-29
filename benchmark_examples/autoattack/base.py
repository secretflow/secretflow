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

from secretflow import PYU


class AutoBase(ABC):
    """
    Abstract base class for autoattack benchmark application, attack, defense case.
    args:
        config (dict): Configuration dict for tune.
        alice: (PYU): alice party.
        bob: (PYU): bob party.
    """

    alice: PYU
    bob: PYU

    def __init__(
        self,
        alice=None,
        bob=None,
    ):
        self.alice = alice
        self.bob = bob
        self.config = None

    @abstractmethod
    def set_config(self, config: Dict[str, str] | None):
        """
        Tuner start a trail with a specific configuration.
        set_config() will be called inside the tunning process.
        Args:
            config (dict): Configuration dict for tune.

        """
        self.config = config if config else {}

    def __enter__(self):
        """All implementations use with to control their resource"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True if exc_type is None else False

    @abstractmethod
    def __str__(self):
        """For better logging, give a readable name of the implementation."""
        pass

    @abstractmethod
    def search_space(self) -> Dict[str, List]:
        """The search space for tuner."""
        pass

    @abstractmethod
    def tune_metrics(self, **kwargs) -> Dict[str, str]:
        """
        The metric name and evaluation method that you want to obtain
        the best results from the tunning results, such as {'auc':'max', 'loss':'min'}.
        For example:
            In ApplicationBase, you may want the max auc of all trails, return {'app_auc':'max'}
                Be sure to add the app_ prefix to distinguish other metrics from attacks.
            In AttackBase, you may want the max acc of attack of all trails, return {'acc':'max'}
        Ensure that the metric names are consistent with the experimental metric names.
        Returns:
            Dict[str, str]: Metrics with name and mode, such as {'auc':'max', 'acc':'max'}
            If no metrics were found, return {}.
        """
        pass
