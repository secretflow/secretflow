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
from typing import Any, Callable, Dict, List, Optional

import ray.tune as tune
from ray.air import RunConfig

import secretflow.distributed as sfd
from secretflow.device import global_state

from .result_grid import ResultGrid
from .tune_config import TuneConfig


class Tuner:
    """
    The secretflow Tuner for launching hyperparameter tuning jobs.
    Args:
        trainable: Then trainable to be tuned.
        cluster_resources[List[Dict]]: The resources for each experiment to use. See example for more information.
        param_space: Search space of the tuning job.
        tune_config: Tuning algorithm specific configs.
        run_config: Runtime configuration that is specific to individual trials.
            If passed, this will overwrite the run config passed to the Trainer,
            if applicable. Refer to ray.air.config.RunConfig for more info.
    Basic usage:

    .. code-block:: python

        import secretflow as sf
        from secretflow import tune
        sf.init(parties=['alice','bob'], address='local')
        def trainable(config):
            alice = sf.PYU('alice')
            bob = sf.PYU('bob')
            # do anything

        tuner = tune.Tuner(
            trainable,
            param_space={'a': tune.grid_search([1,2,3])}
        )

    Each experiment in the above example will comsume all resources and be executed serially.
    You can  manually specify the resource usage in each experiment to achive optimal parallelism and performance.

    .. code-block:: python

        tuner = tune.Tuner(
            trainable,
            cluster_resources=[
                {'alice': 1, 'CPU' 4},
                {'bob': 1, 'CPU' 4}
            ]
            param_space = {'a': tune.grid_search([1,2,3])}
        )
    In the above example, both alice and bob will use 4 CPU in each experiment.
    If your machine has 16 CPUs, Tune will run 2 experiments in parallel.
    Note that the numbers associated with PYU (above, Alice or Bob which is 1)
    have no significance and can be any value,
    but the custom resources you define need to be set correct.
    """

    ray_tune: tune.Tuner

    def __init__(
        self,
        trainable: Callable = None,
        cluster_resources: Optional[List[Dict]] = None,
        *,
        param_space: Optional[Dict[str, Any]] = None,
        tune_config: Optional[TuneConfig] = None,
        run_config: Optional[RunConfig] = None,
    ):
        tune_resources = self._init_resources(cluster_resources)
        trainable = tune.with_resources(trainable, resources=tune_resources)
        self.ray_tune = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )

    def _init_resources(self, cluster_resources):
        avaliable_resources = sfd.get_cluster_avaliable_resources()
        if cluster_resources is None:
            logging.warning(
                f"Tuner() got arguments cluster_resources=None. "
                f"The Tuner will defaultly use as many as cluster resources in each experiment."
                f"That means each experiments of tune will occupy all the machine's resources and "
                f"experiments can only execute serial."
                f"To achieve better tuning performance, please refer to the cluster_resources arguments "
                f"and control the resources by yourself.."
            )
            parties = global_state.parties()
            cluster_resources = []
            for party in parties:
                party_resources = {party: avaliable_resources[party] - 2}
                for res_name, avalia in avaliable_resources.items():
                    if (
                        res_name not in parties
                        and 'memory' not in res_name
                        and 'node' not in res_name
                    ):
                        party_resources[res_name] = int(avalia / len(parties)) - 1
                cluster_resources.append(party_resources)
        self._check_resources_input(cluster_resources, avaliable_resources)
        logging.warning(f"cluster_resources = {cluster_resources}")
        tune_resources = tune.PlacementGroupFactory(
            [
                {},  # the trainanble itself in sf will not use any resources
                *cluster_resources,
            ],
            strategy="PACK",
        )
        return tune_resources

    @staticmethod
    def _check_resources_input(
        cluster_resources: List[Dict], avaliable_resources: Dict
    ):
        total_resources = {}
        for cluster_resource in cluster_resources:
            for name, value in cluster_resource.items():
                if name in total_resources:
                    total_resources[name] += value
                else:
                    total_resources[name] = value
        for name, value in total_resources.items():
            assert name in avaliable_resources, f"Got unknown resource name {name}"
            assert (
                value <= avaliable_resources[name]
            ), f"Total resouces {name}:{value} required by user exceed the avaliable resources {avaliable_resources[name]}"

    def fit(self) -> ResultGrid:
        return self.ray_tune.fit()
