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
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import ray
import ray.tune as tune
from ray.air import RunConfig

import secretflow.distributed as sfd
from secretflow.device import global_state

from ..distributed.primitive import DISTRIBUTION_MODE
from .result_grid import ResultGrid
from .tune_config import TuneConfig


def trainable_wrapper(trainable: Callable, global_params: Dict):
    """
    Wrapper function for trainable functions.
    This function is used to launch hyperparameter tuning jobs.
    Args:
        trainable: Then trainable to be tuned.
        global_params: Global parameters for the trainable function.
    Returns:
        A dictionary containing the results of the training.
    """

    @wraps(trainable)
    def wrapper(config: Dict):
        if 'distribution_mode' in global_params:
            sfd.set_distribution_mode(global_params['distribution_mode'])
        return trainable(config)

    return wrapper


class Tuner:
    """
    The secretflow Tuner for launching hyperparameter tuning jobs.
    Args:
        trainable: Then trainable to be tuned.
        cluster_resources[List[Dict], Dict]: The resources for each experiment to use. See example for more information.
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

    If using debug mode, all devices run in the same process.
    You can specify a Dict resources usage like:

    .. code-block:: python

        tuner = tune.Tuner(
            trainable,
            cluster_resources= {'alice': 1, 'bob': 1, 'CPU' 4},
            param_space = {'a': tune.grid_search([1,2,3])}
        )

    In this example, one trail will consume 1 resource each for Alice and Bob,
    with a total of 4 CPUs being used.

    When using sim mode, different devices run in separate processes,
    and we need to specify the resource usage for each worker.

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

    Note that List input can also work in debug mode, the program will consider the total
    sum of all resources in the list as the resources used by one trail.
    """

    ray_tune: tune.Tuner

    def __init__(
        self,
        trainable: Callable = None,
        cluster_resources: Optional[
            Union[List[Dict[str, float]], Dict[str, float]]
        ] = None,
        *,
        param_space: Optional[Dict[str, Any]] = None,
        tune_config: Optional[TuneConfig] = None,
        run_config: Optional[RunConfig] = None,
    ):
        trainable = self._handle_global_params(trainable)
        trainable = self._construct_trainable_with_resources(
            trainable, cluster_resources
        )
        self.ray_tune = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )

    @staticmethod
    def _handle_global_params(traiable):
        """
        sf.init will set some global parameters in driver side,
        which will not be passed to worker when use tuner to tune sf.
        So we need to reset these paraeters again.
        This function will wrapper the global parameters and set them.
        """
        global_params = {'distribution_mode': sfd.get_distribution_mode()}
        return trainable_wrapper(traiable, global_params)

    def _construct_trainable_with_resources(self, trainable, cluster_resources):
        distribution_mode = sfd.get_distribution_mode()
        if distribution_mode == DISTRIBUTION_MODE.DEBUG:
            tune_resources = self._init_debug_resources(cluster_resources)
        elif distribution_mode == DISTRIBUTION_MODE.SIMULATION:
            tune_resources = self._init_sim_resources(cluster_resources)
        else:
            raise NotImplementedError()

        return tune.with_resources(trainable, resources=tune_resources)

    def _init_debug_resources(self, cluster_resources):
        if not ray.is_initialized():
            logging.warning(
                "When using the debug mode, "
                "the Tuner will automatically calls ray.init() without any parameters to start Ray. "
                "This doesn't include any resource parameters. "
                "If you want to control the resources, "
                "please try starting the Ray cluster manually "
                "or directly call ray.init() on a single machine and provide appropriate parameters."
            )
            ray.init()
        avaliable_resources = ray.available_resources()
        for party in global_state.parties():
            if party not in avaliable_resources:
                avaliable_resources[party] = avaliable_resources['CPU']
        if cluster_resources is None or isinstance(cluster_resources, List):
            if isinstance(cluster_resources, List):
                logging.warning(
                    "Tuner suggest a Dict cluster_resources input but got List."
                    "It will be transformed into a Dict with its sums."
                    f"{cluster_resources}"
                )
            else:
                cluster_resources = self._default_cluster_resource(avaliable_resources)
            resources = {}
            for res in cluster_resources:
                for k, v in res.items():
                    if k not in resources:
                        resources[k] = v
                    else:
                        resources[k] += v
            cluster_resources = resources

        self._check_resources_input(cluster_resources, avaliable_resources)
        return cluster_resources

    def _init_sim_resources(self, cluster_resources):
        avaliable_resources = ray.available_resources()
        if cluster_resources is None:
            cluster_resources = self._default_cluster_resource(avaliable_resources)
        self._check_resources_input(cluster_resources, avaliable_resources)
        tune_resources = tune.PlacementGroupFactory(
            [
                {},  # the trainanble itself in sf will not use any resources
                *cluster_resources,
            ],
            strategy="PACK",
        )
        return tune_resources

    @staticmethod
    def _default_cluster_resource(avaliable_resources):
        logging.warning(
            f"Tuner() got arguments cluster_resources=None. "
            f"The Tuner will defaultly use as many as cluster resources in each experiment."
            f"That means each experiments of tune will try to occupy all the machine's resources and "
            f"experiments can only be executed serial."
            f"To achieve better tuning performance, please refer to the cluster_resources arguments "
            f"and control the resources by yourself."
        )
        parties = global_state.parties()
        cluster_resources = []
        # When use sart ray cluster with node nums > 1, we cannot run a trail coross differnt nodes,
        #  so we should keep the resource usage less than it in one node.
        node_nums = 1
        for k in avaliable_resources:
            if "node" in k:
                node_nums += 1
        for party in parties:
            party_resources = {
                party: max(avaliable_resources[party] / node_nums - 2, 1)
            }
            for res_name, avalia in avaliable_resources.items():
                if (
                    res_name not in parties
                    and 'memory' not in res_name
                    and 'node' not in res_name
                ):
                    party_resources[res_name] = max(
                        int(avalia / node_nums / len(parties)) - 1, 1
                    )
            cluster_resources.append(party_resources)
        return cluster_resources

    @staticmethod
    def _check_resources_input(
        cluster_resources: Union[List[Dict], Dict], avaliable_resources: Dict
    ):
        if isinstance(cluster_resources, Dict):
            cluster_resources = [cluster_resources]
        columns = (
            ['avaliable']
            + ["res_worker_" + str(i) for i in range(len(cluster_resources))]
            + ['res_per_trail']
        )
        resource_usage = OrderedDict(
            {
                k: OrderedDict(
                    {name: v if name == 'avaliable' else 0 for name in columns}
                )
                for k, v in avaliable_resources.items()
            }
        )
        # record all resource avaliable and each worker usage.
        for i, cluster_resource in enumerate(cluster_resources):
            for r, v in cluster_resource.items():
                assert (
                    r in resource_usage
                ), f"Got unknown resource name {r}, avaliable names contains {avaliable_resources.keys()}"
                resource_usage[r]["res_worker_" + str(i)] = v
                resource_usage[r]['res_per_trail'] += v
        # check if the usage exeeds the avaliable limit.
        for r, v in resource_usage.items():
            assert (
                v['res_per_trail'] <= v['avaliable']
            ), f"Total resouces {r}:{v['res_per_trail']} required by user exceed the avaliable resources {v['avaliable']}"
        # show the avaliable and the requirment of resources for each trail.
        resource_record = {'resource': [r for r in resource_usage.keys()]}
        resource_record.update(
            {k: [str(r[k]) for r in resource_usage.values()] for k in columns}
        )
        resource_record = pd.DataFrame(resource_record).astype(str)
        # show mem in xx.x MB
        for i in range(resource_record.shape[0]):
            if 'mem' in resource_record['resource'][i]:
                for j in range(1, resource_record.shape[1]):
                    resource_record.iloc[i, j] = (
                        str(round(float(resource_record.iloc[i, j]) / 1024 / 1024, 1))
                        + "MB"
                    )
        logging.warning(
            f"The utilization of the Tuner resources is as follows:\n{resource_record.to_markdown()}"
        )

    def fit(self) -> ResultGrid:
        return self.ray_tune.fit()
