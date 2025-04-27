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

from typing import Callable, Dict, List


class ResourceDict(dict):
    """
    A dictionary used to indicate resource consumption for applications.
    Args:
        gpu_mem: The amount of GPU memory needed. Generally, this is the most important value
                for resource allocation.
        CPU: The number of CPUs used, default is 1.
        memory: Memory consumption. If the application don't need a large amount of storage,
                you can use the default value of 0.
        **kwargs: Other user defined values. Please note that if you use SIM mode,
                you will need to fill in the participation method here. e.g. alice = 0.01.
    """

    def __init__(self, gpu_mem: float, CPU: float = 1, memory: float = 0, **kwargs):
        super().__init__(
            CPU=CPU,
            memory=memory,
            gpu_mem=gpu_mem,
            **kwargs,
        )

    def without_gpu(self):
        """Get dict without keys relevant to GPU."""
        return {k: v for k, v in self.items() if k not in ['GPU', 'gpu_mem']}

    def handle_gpu_mem(
        self, gpu_config: dict
    ) -> Dict[str, float] | List[Dict[str, float]]:
        """
        Translate the GPU memry value into percentage of GPU
        and return the cluster resources recognized by SfTuner.

        The primary reason to translate is that Ray does not directly provide metrics for managing GPU memory.
        Instead, it only indicates GPU memory usage through a percentage of the tag 'GPU',
        which fails to work properly in scenarios where multiple GPUs with differing memory capacities are present.

        The main idea of this function is adding a GPU type field (accelerator_type) into resource labels
        and generating distinct GPU labels based on the memory configurations of different GPU types.
        Consequently, when Ray allocates tasks to a specific GPU, the associated GPU percentage label
        accurately represents the GPU memory usage for that allocation. Reffer to:
        https://discuss.ray.io/t/tune-sgd-rllib-distribute-training-across-nodes-with-different-gpus/1522/5

        For example, suppose you have one GPU of each type: a 4GB V100 and an 8GB A100.
        Your configuration should be {'V100': 4000000000, 'A100': 8000000000}.
        Suppose each trail intends to utilize 2GB GPU memory units.
        After conversion, this process generates two candidate resource labels:
        {'accelerator_type:V100': 0.001, 'GPU': '0.5'} and {'accelerator_type:A100': 0.001, 'GPU': '0.25'}.
        The SFTuner will recognize these candidates and randomly assign one for use.

        Args:
            gpu_config: The configuration of GPU such as {'V100': 4000000000, 'A100': 8000000000}

        Returns:
            A dict contains all the resources or a List of dicts contains all the possibal resources.
        """
        assert len(gpu_config) > 0
        if len(gpu_config) == 1:
            # only one type of gpu
            total_mem_per_gpu = next(iter(gpu_config.values()))
            self['GPU'] = self.pop('gpu_mem') / total_mem_per_gpu
            return self
        else:
            candidates = []
            for gpu_type, total_mem_per_gpu in gpu_config.items():
                my_self = self.copy()
                my_self.pop('gpu_mem')
                gpu_percentage = self['gpu_mem'] / total_mem_per_gpu
                my_self['GPU'] = gpu_percentage
                my_self[f'accelerator_type:{gpu_type}'] = 0.001
                candidates.append(my_self)
            return candidates


class ResourcesPack:
    """
    A Pack of:
        - resources consumptions for debug_mode.
        - each party's resource consumptions for simulation_mode.
    """

    def __init__(self):
        self.resources: Dict[str, ResourceDict] = {}

    def with_debug_resources(self, r: ResourceDict) -> 'ResourcesPack':
        # append debug resources
        assert isinstance(r, ResourceDict)
        self.resources['debug'] = r
        return self

    def with_sim_resources(self, party: str, r: ResourceDict) -> 'ResourcesPack':
        # append a party's sim resources
        assert isinstance(r, ResourceDict)
        self.resources[party] = r
        return self

    def apply_debug_resources(
        self, key: str, func: Callable[[float], float]
    ) -> 'ResourcesPack':
        # update debug resources with the key and target function
        assert (
            'debug' in self.resources
        ), "need with_debug_resources before do anything."
        assert key in self.resources['debug'], f"{key} not in resources."
        self.resources['debug'][key] = func(self.resources['debug'][key])
        return self

    def apply_sim_resources(
        self, party: str, key: str, func: Callable[[float], float]
    ) -> 'ResourcesPack':
        # update debug resources with the party name, key, and target function
        assert party in self.resources, "need with_sim_resources before do anything."
        assert key in self.resources['debug'], f"{key} not in resources."
        self.resources[party][key] = func(self.resources[party][key])
        return self

    def get_debug_resources(self) -> List[ResourceDict]:
        assert (
            'debug' in self.resources
        ), "need with_debug_resources before do anything."
        return [self.resources['debug']]

    def get_all_sim_resources(self) -> List[ResourceDict]:
        if 'debug' in self.resources:
            self.resources.pop('debug')
        assert len(self.resources) > 1, f'Need >1 resources consumptions parties.'
        return list(self.resources.values())
