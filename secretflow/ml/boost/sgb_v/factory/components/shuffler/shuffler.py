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

from dataclasses import dataclass
from typing import List, Union

from secretflow.device import PYUObject
from secretflow.ml.boost.sgb_v.core.params import default_params
from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

from ..component import Component, Devices, print_params
from .worker_shuffler import WorkerShuffler


@dataclass
class ShufflerParams:
    """
    'seed': Pseudorandom number generator seed.
        default: 1212
    """

    seed: int = default_params.seed


class Shuffler(Component):
    def __init__(self):
        self.worker_shufflers = []
        self.params = ShufflerParams()

    def show_params(self):
        print_params(self.params)

    def set_params(self, params: dict):
        self.params.seed = params.get('seed', default_params.seed)

    def get_params(self, params: dict):
        params['seed'] = self.params.seed

    def set_devices(self, devices: Devices):
        self.workers = devices.workers
        return

    def set_actors(self, actors: List[SGBActor]):
        self.worker_shufflers = actors
        for worker in self.worker_shufflers:
            # may change random state initialie methods latter
            worker.register_class('WorkerShuffler', WorkerShuffler, self.params.seed)

    def del_actors(self):
        del self.worker_shufflers

    def reset_shuffle_masks(self):
        for ws in self.worker_shufflers:
            ws.invoke_class_method('WorkerShuffler', 'reset_shuffle_mask')

    def create_shuffle_mask(
        self, worker_index: int, key: int, bucket_list: List[PYUObject]
    ) -> List[int]:
        return self.worker_shufflers[worker_index].invoke_class_method(
            'WorkerShuffler', 'create_shuffle_mask', key, bucket_list
        )

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
            worker_shuffler.invoke_class_method(
                'WorkerShuffler', 'undo_shuffle_mask_list_wise', split_buckets
            )
            for worker_shuffler, split_buckets in zip(
                self.worker_shufflers, split_buckets_parition_wise
            )
        ]

    def unshuffle_split_buckets_with_keys(
        self,
        split_buckets_parition_wise: List[PYUObject],
        keys: Union[PYUObject, List[PYUObject]],
    ) -> List[PYUObject]:
        """unshuffle split buckets viewed by each parition

        Args:
            split_buckets_parition_wise (List[PYUObject]): PYUObject is List[int], split buckets viewed from this partition
            keys (Union[PYUObject, List[PYUObject]]): Both cases are in fact List[int]. keys will be sent to all workers

        Returns:
            List[List[PYUObject]]: unshuffled split buckets
        """
        return [
            worker_shuffler.invoke_class_method(
                'WorkerShuffler',
                'undo_shuffle_mask_with_keys',
                split_buckets,
                keys.to(worker_shuffler.device)
                if isinstance(keys, PYUObject)
                else [k.to(worker_shuffler.device) for k in keys],
            )
            for worker_shuffler, split_buckets in zip(
                self.worker_shufflers, split_buckets_parition_wise
            )
        ]
