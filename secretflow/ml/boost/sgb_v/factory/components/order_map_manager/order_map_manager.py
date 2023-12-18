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

import math
from dataclasses import dataclass
from typing import Dict, List, Union

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device import PYUObject
from secretflow.ml.boost.sgb_v.core.params import default_params
from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

from ..component import Component, Devices, print_params
from ..logging import LoggingParams, LoggingTools
from .order_map_actor import OrderMapActor


@dataclass
class OrderMapBuilderParams:
    """
    'sketch_eps': This roughly translates into O(1 / sketch_eps) number of bins.
        default: 0.1
        range: (0, 1]

    'seed': Pseudorandom number generator seed.
        default: 1212
    """

    sketch_eps: float = default_params.sketch_eps
    seed: int = default_params.seed


class OrderMapManager(Component):
    def __init__(self) -> None:
        self.params = OrderMapBuilderParams()
        self.logging_params = LoggingParams()
        self.buckets = eps_inverse(self.params.sketch_eps)
        self.order_map_actors = []

    def show_params(self):
        print_params(self.params)
        print_params(self.logging_params)

    def set_params(self, params: Dict):
        # validate
        sketch = params.get('sketch_eps', default_params.sketch_eps)
        self.params.sketch_eps = sketch
        # derive attributes
        self.buckets = eps_inverse(sketch)
        self.params.seed = params.get('seed', default_params.seed)

        self.logging_params = LoggingTools.logging_params_from_dict(params)

    def get_params(self, params: dict):
        params['sketch_eps'] = self.params.sketch_eps
        params['seed'] = self.params.seed

        LoggingTools.logging_params_write_dict(params, self.logging_params)

    def set_devices(self, devices: Devices):
        self.workers = devices.workers

    def set_actors(self, actors: SGBActor):
        self.order_map_actors = actors
        for i, actor in enumerate(self.order_map_actors):
            actor.register_class('OrderMapActor', OrderMapActor, i)

    def del_actors(self):
        del self.order_map_actors

    @LoggingTools.enable_logging
    def build_order_map(self, x: FedNdarray) -> FedNdarray:
        # we assumed x's devices match when setting up devices.
        buckets, seed = self.buckets, self.params.seed
        self.order_map = FedNdarray(
            {
                order_map_actor.device: order_map_actor.invoke_class_method(
                    'OrderMapActor',
                    'build_order_map',
                    x.partitions[order_map_actor.device].data,
                    buckets,
                    seed,
                )
                for order_map_actor in self.order_map_actors
            },
            partition_way=PartitionWay.VERTICAL,
        )
        return self.order_map

    def get_order_map(self) -> FedNdarray:
        return self.order_map

    def get_feature_buckets(self) -> List[PYUObject]:
        return [
            order_map_actor.invoke_class_method('OrderMapActor', 'get_feature_buckets')
            for order_map_actor in self.order_map_actors
        ]

    def get_bucket_lists(self, col_choices_list: List[PYUObject]) -> List[PYUObject]:
        return [
            self.order_map_actors[i].invoke_class_method(
                'OrderMapActor', 'get_bucket_list', col_choices
            )
            for i, col_choices in enumerate(col_choices_list)
        ]

    def compute_left_child_selects(
        self,
        actor_index: int,
        feature: int,
        split_point_index: int,
        sampled_indices: Union[List[int], None] = None,
    ) -> PYUObject:
        return self.order_map_actors[actor_index].invoke_class_method(
            'OrderMapActor',
            'compute_left_child_selects',
            feature,
            split_point_index,
            sampled_indices,
        )

    def batch_query_split_points_each_party(
        self, queries_list: List[PYUObject]
    ) -> List[PYUObject]:
        return [
            actor.invoke_class_method(
                'OrderMapActor', 'batch_query_split_points', queries
            )
            for actor, queries in zip(self.order_map_actors, queries_list)
        ]

    def batch_compute_left_child_selects_each_party(
        self,
        split_feature_buckets_each_party: List[PYUObject],
        sampled_indices: Union[List[int], None] = None,
    ) -> List[PYUObject]:
        return [
            actor.invoke_class_method(
                'OrderMapActor',
                'batch_compute_left_child_selects',
                queries,
                sampled_indices,
            )
            for actor, queries in zip(
                self.order_map_actors, split_feature_buckets_each_party
            )
        ]


def eps_inverse(eps):
    return math.ceil(1.0 / eps)
