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

import itertools
import logging
import random
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import ray
import ray.tune as tune
from ray.air import RunConfig
from ray.tune import PlacementGroupFactory

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
        cluster_resources: The resources for each experiment to use. See example for more information.
            If it is None, then generate a default cluster_resources it generates default cluster resources,
                with each experiment consuming all available resources.
            If it is a Dict[str, float], or a List[Dict[str,float]] with length 1, and the debug_mode is enabled,
                It will be used as the consumption for each experiment.
            If it is a List[Dict[str,float]], and the debug_mode is disabled, the elements within represent the
                resources consumed by each remote worker created during the experiment.
            If it is a List[List[Dict[str,float]]], then each inner list represents an option for resource consumption.
                In this case, every experiment will randomly select one of these options to determine its resource usage.
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
            cluster_resources= {'alice': 1, 'bob': 1, 'CPU' 4}, # or [{'alice': 1, 'bob': 1, 'CPU' 4}]
            param_space = {'a': tune.grid_search([1,2,3])}
        )

    In this example, each experiment will consume 1 alice, 1 bob and 4 CPUs,

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

    In the above example, both alice and bob will use 4 CPUs in each experiment.
    If your machine has 16 CPUs, Tune will run 2 experiments in parallel.

    If you have several options for resources and wish to select among them randomly,
    you can directly specify a range of alternatives.
    A common scenario being assigning different resources based on varying GPU types,
    suppose you have one GPU of each type: a 4GB V100 and an 8GB A100.
    Suppose each experiment intends to utilize 2GB GPU memory units.
    Since you can only indicate gpu memory by the percentage of 'GPU' label,
    You can use following option resouces:

    .. code-block:: python

        # if enable debug_mode
        tuner = tune.Tuner(
            trainable,
            cluster_resources=[
                [
                    {accelerator_type:V100, CPU:1, GPU: 0.5}, # use 4 * 0.5 = 2GB
                    {accelerator_type:A100, CPU:1, GPU:0.25} # use 8 * 0.25 = 2GB
                ],
            ]
            param_space = {'a': tune.grid_search([1,2,3])}
        )

        # if enable sim_mode
        tuner = tune.Tuner(
            trainable,
            cluster_resources=[
                [
                    {alice:1, accelerator_type:V100, CPU:1, GPU: 0.25}, # use 4 * 0.25 = 1GB
                    {alice:1, accelerator_type:A100, CPU:1, GPU: 0.25}, # use 8 * 0.5 = 1GB
                ],
                [
                    {bob:1, accelerator_type:V100, CPU:1, GPU: 0.25}, # use 4 * 0.25 = 1GB
                    {bob:1, accelerator_type:A100, CPU:1, GPU: 0.25}, # use 8 * 0.5 = 1GB
                ]
            ]
            param_space = {'a': tune.grid_search([1,2,3])}
        )
    """

    ray_tune: tune.Tuner

    def __init__(
        self,
        trainable: Callable = None,
        cluster_resources: (
            None
            | Dict[str, float]
            | List[Dict[str, float]]
            | List[List[Dict[str, float]]]
        ) = None,
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

    def _construct_trainable_with_resources(
        self,
        trainable,
        cluster_resources: (
            None
            | Dict[str, float]
            | List[Dict[str, float]]
            | List[List[Dict[str, float]]]
        ) = None,
    ):
        distribution_mode = sfd.get_distribution_mode()
        if distribution_mode == DISTRIBUTION_MODE.DEBUG:
            tune_resources = self._init_debug_resources(cluster_resources)
        elif distribution_mode == DISTRIBUTION_MODE.SIMULATION:
            tune_resources = self._init_sim_resources(cluster_resources)
        else:
            raise NotImplementedError()
        return tune.with_resources(trainable, resources=tune_resources)

    def _init_debug_resources(
        self,
        cluster_resources: (
            None
            | Dict[str, float]
            | List[Dict[str, float]]
            | List[List[Dict[str, float]]]
        ) = None,
    ):
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
        if cluster_resources is None:
            cluster_resources = self._default_cluster_resource(
                avaliable_resources, is_debug=True
            )
        self._check_resources_input(cluster_resources, avaliable_resources)
        if isinstance(cluster_resources, List):
            if len(cluster_resources) != 1:
                raise ValueError(
                    "When using debugging mode, Tuner requires a resource usage to be "
                    "either dict format or a list format with a length of 1, "
                    "as each experiment only requires starting one worker"
                )
            cluster_resource = cluster_resources[0]
            if isinstance(cluster_resource, List):
                # https://discuss.ray.io/t/tune-sgd-rllib-distribute-training-across-nodes-with-different-gpus/1522/5
                cluster_resources = lambda _: PlacementGroupFactory(
                    [random.choice(cluster_resource)]
                )
            else:
                # single gpu type
                cluster_resources = cluster_resource
        return cluster_resources

    def _init_sim_resources(
        self,
        cluster_resources: (
            None
            | Dict[str, float]
            | List[Dict[str, float]]
            | List[List[Dict[str, float]]]
        ) = None,
    ):
        avaliable_resources = ray.available_resources()
        if cluster_resources is None:
            cluster_resources = self._default_cluster_resource(avaliable_resources)
        elif isinstance(cluster_resources, Dict):
            raise ValueError(
                "When using debugging mode, Tuner requires a resource usage to be "
                "a list format with the length of the nums of workers per trail. "
            )
        self._check_resources_input(cluster_resources, avaliable_resources)
        if any([isinstance(cr, List) for cr in cluster_resources]):
            # multi choice
            tune_resources = lambda _: PlacementGroupFactory(
                [
                    {},
                    *[random.choice(cr) for cr in cluster_resources],
                ],
                strategy="PACK",
            )
        else:
            # single choice
            tune_resources = tune.PlacementGroupFactory(
                [
                    {},  # the trainanble itself in sf will not use any resources
                    *cluster_resources,
                ],
                strategy="PACK",
            )
        return tune_resources

    def _default_cluster_resource(
        self, avaliable_resources, is_debug=False
    ) -> List[Dict[str, float]]:
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
        # When use sart ray cluster with node nums > 1, we cannot run one trail coross differnt nodes,
        #  so we should keep the resource usage less than it in one node.
        node_nums = len(ray.nodes())
        if is_debug:
            trail_resources = {
                res_name: avalia
                for res_name, avalia in avaliable_resources.items()
                if self._is_consumable_resource(res_name)
            }
            cluster_resources.append(trail_resources)
        else:
            for party in parties:
                party_resources = {
                    party: max(avaliable_resources[party] / node_nums, 0)
                }
                for res_name, avalia in avaliable_resources.items():
                    if (
                        self._is_consumable_resource(res_name)
                        and res_name not in parties
                    ):
                        party_resources[res_name] = max(
                            avalia / node_nums / len(parties), 0
                        )
                cluster_resources.append(party_resources)
        return cluster_resources

    @staticmethod
    def _is_consumable_resource(resource_name: str) -> bool:
        """Check if the resource name is a consumable resource."""
        if (
            'node:' not in resource_name
            and 'accelerator_type' not in resource_name
            and resource_name != 'memory'
            and resource_name != 'object_store_memory'
        ):
            return True
        return False

    @staticmethod
    def _check_resources_input(
        cluster_resources: (
            Dict[str, float] | List[Dict[str, float]] | List[List[Dict[str, float]]]
        ),
        avaliable_resources: Dict,
    ):
        if isinstance(cluster_resources, Dict):
            cluster_resources = [cluster_resources]
        if isinstance(cluster_resources, List) and any(
            [isinstance(cr, List) for cr in cluster_resources]
        ):
            for valid_cr in itertools.product(*cluster_resources):
                Tuner._check_resources_input(list(valid_cr), avaliable_resources)
            return
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
        missing_resources = []
        for i, cluster_resource in enumerate(cluster_resources):
            for r, v in cluster_resource.items():
                if r not in resource_usage:
                    missing_resources.append(r)
                    continue
                resource_usage[r]["res_worker_" + str(i)] = v
                resource_usage[r]['res_per_trail'] += v
        if len(missing_resources) > 0:
            err_msg = f"Got unknown required resources {missing_resources}, avaliable names contains {avaliable_resources.keys()}. "
            if len([... for ms in missing_resources if 'accelerator_type' in ms]) > 0:
                # user defined resources contains gpu accelerator_type
                err_msg += f"When using GPUs, make sure you set the correct type name of your gpus in your config file. "
                avali_accelerator_types = [
                    rs for rs in avaliable_resources if 'accelerator_type' in rs
                ]
                if len(avali_accelerator_types) == 0:
                    err_msg += "Current env does not have any gpu, please check your env or ray start."
                else:
                    err_msg += (
                        f"Current env have gpu with type {avali_accelerator_types}"
                    )
            raise ValueError(err_msg)
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
