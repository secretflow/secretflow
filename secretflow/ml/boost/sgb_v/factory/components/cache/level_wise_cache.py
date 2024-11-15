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
from typing import List

from ....core.cache.level_cache import LevelCache
from ..component import Component, Devices


class LevelWiseCache(Component):
    def __init__(self):
        self.worker_level_caches = {}
        self.workers = []

    def show_params(self):
        return

    def set_params(self, _: dict):
        return

    def get_params(self, _: dict):
        return

    def set_devices(self, devices: Devices):
        self.workers = devices.workers
        self.label_holder = devices.label_holder

    def set_actors(self, _):
        self.worker_level_caches = {worker: LevelCache() for worker in self.workers}

    def del_actors(self):
        del self.worker_level_caches

    def reset_level_caches(self):
        for device in self.worker_level_caches.keys():
            self.worker_level_caches[device].reset_level_nodes_GH()

    def collect_level_node_GH(self, worker, bucket_sums, is_lefts):
        self.worker_level_caches[worker].collect_level_node_GH_level_wise(
            bucket_sums, is_lefts
        )

    def get_level_nodes_GH(self, worker) -> List:
        return self.worker_level_caches[worker].get_level_nodes_GH()

    def update_level_cache(self, is_last_level, gain_is_cost_effective):
        for cache in self.worker_level_caches.values():
            cache.update_level_cache(is_last_level, gain_is_cost_effective)

    def reset(self):
        for cache in self.worker_level_caches.values():
            cache.reset()
