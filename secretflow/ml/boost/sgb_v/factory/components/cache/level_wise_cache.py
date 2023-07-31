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

from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

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

    def set_actors(self, actors: SGBActor):
        # HEUObjects cannot be put into actors, we have to cache it at driver
        self.worker_level_caches = {
            worker: LevelCache()
            for worker in self.workers
            if worker != self.label_holder
        }
        for actor in actors:
            if actor.device == self.label_holder:
                self.worker_level_caches[self.label_holder] = actor
                break
        self.worker_level_caches[self.label_holder].register_class(
            'LevelCache', LevelCache
        )

    def del_actors(self):
        del self.worker_level_caches

    def reset_level_caches(self):
        for device in self.worker_level_caches.keys():
            if device != self.label_holder:
                self.worker_level_caches[device].reset_level_nodes_GH()
            else:
                self.worker_level_caches[device].invoke_class_method(
                    'LevelCache', 'reset_level_nodes_GH'
                )

    def collect_level_node_GH(self, worker, bucket_sums, is_lefts):
        if worker != self.label_holder:
            self.worker_level_caches[worker].collect_level_node_GH_level_wise(
                bucket_sums, is_lefts
            )
        else:
            self.worker_level_caches[worker].invoke_class_method(
                'LevelCache', 'collect_level_node_GH_level_wise', bucket_sums, is_lefts
            )

    def get_level_nodes_GH(self, worker) -> List:
        if worker != self.label_holder:
            return self.worker_level_caches[worker].get_level_nodes_GH()
        else:
            return self.worker_level_caches[worker].invoke_class_method(
                'LevelCache', 'get_level_nodes_GH'
            )

    def update_level_cache(self, is_last_level, gain_is_cost_effective):
        for worker, cache in self.worker_level_caches.items():
            if worker != self.label_holder:
                cache.update_level_cache(is_last_level, gain_is_cost_effective)
            else:
                cache.invoke_class_method(
                    'LevelCache',
                    'update_level_cache',
                    is_last_level,
                    gain_is_cost_effective,
                )

    def reset(self):
        for worker, cache in self.worker_level_caches.items():
            if worker != self.label_holder:
                cache.reset()
            else:
                cache.invoke_class_method('LevelCache', 'reset')
