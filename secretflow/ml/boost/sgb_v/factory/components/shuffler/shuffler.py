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

from dataclasses import dataclass
from secretflow.device import PYUObject

from .worker_shuffler import WorkerShuffler
from ..component import Component, Devices, print_params


@dataclass
class ShufflerParams:
    """
    'seed': Pseudorandom number generator seed.
        default: 1212
    """

    seed: int = 1212


class Shuffler(Component):
    def __init__(self):
        self.worker_shufflers = []
        self.params = ShufflerParams()

    def show_params(self):
        print_params(self.params)

    def set_params(self, params: dict):
        self.params.seed = int(params.get('seed', 1212))

    def get_params(self, params: dict):
        params['seed'] = self.params.seed

    def set_devices(self, devices: Devices):
        self.worker_shufflers = [
            WorkerShuffler(self.params.seed, device=worker)
            for worker in devices.workers
        ]
        return

    def reset_shuffle_masks(self):
        for ws in self.worker_shufflers:
            ws.reset_shuffle_mask()

    def create_shuffle_mask(
        self, worker_index: int, key: int, bucket_list: List[PYUObject]
    ) -> List[int]:
        return self.worker_shufflers[worker_index].create_shuffle_mask(key, bucket_list)

    def unshuffle_split_buckets(
        self, split_buckets_parition_wise: List[PYUObject]
    ) -> List[PYUObject]:
        """unshuffle split buckets viewed by each parition

        Args:
            split_buckets_parition_wise (List[PYUObject]): PYUObject is List[int], split buckets viewed from this partition

        Returns:
            List[List[PYUObject]]: unshuffled split buckets
        """
        return [
            worker_shuffler.undo_shuffle_mask_list_wise(split_buckets)
            for worker_shuffler, split_buckets in zip(
                self.worker_shufflers, split_buckets_parition_wise
            )
        ]
