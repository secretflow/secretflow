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
from typing import Dict, List, Tuple

from secretflow.data import FedNdarray
from secretflow.device import PYU, HEUObject, PYUObject
from secretflow.device.device.heu import HEUMoveConfig
from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

from ....core.pure_numpy_ops.bucket_sum import regroup_bucket_sums
from ....core.pure_numpy_ops.grad import split_GH
from ....core.pure_numpy_ops.node_select import (
    packbits_node_selects,
    unpackbits_node_selects,
)
from ..cache.node_wise_bucket_sum_cache import NodeWiseCache
from ..component import (
    Composite,
    Devices,
    print_params,
    set_dict_from_params,
    set_params_from_dict,
)
from ..gradient_encryptor import GradientEncryptor
from ..logging import LoggingParams, LoggingTools
from ..shuffler import Shuffler


@dataclass
class LeafWiseBucketSumCalculatorParams:
    """
    'label_holder_feature_only': bool. if true, non-label holder will not do bucket sum
        default: False
    'enable_packbits': bool. if true, turn on packbits transmission.
        default: False
    """

    label_holder_feature_only: bool = False
    enable_packbits: bool = False


@dataclass
class LeafWiseBucketSumCalculatorComponents:
    node_wise_cache: NodeWiseCache = NodeWiseCache()


class LeafWiseBucketSumCalculator(Composite):
    def __init__(self):
        self.components = LeafWiseBucketSumCalculatorComponents()
        self.logging_params = LoggingParams()
        self.params = LeafWiseBucketSumCalculatorParams()

    def show_params(self):
        print_params(self.logging_params)
        print_params(self.params)

    def set_params(self, params: dict):
        LoggingTools.logging_params_from_dict(params, self.logging_params)
        set_params_from_dict(self.params, params)

    def get_params(self, params: dict):
        LoggingTools.logging_params_write_dict(params, self.logging_params)
        set_dict_from_params(self.params, params)

    def set_devices(self, devices: Devices):
        super().set_devices(devices)
        self.label_holder = devices.label_holder
        self.workers = devices.workers
        self.party_num = len(self.workers)
        self.heu = devices.heu

    def set_actors(self, actors: List[SGBActor]):
        return super().set_actors(actors)

    def del_actors(self):
        return super().del_actors()

    @LoggingTools.enable_logging
    def calculate_bucket_sum(
        self,
        shuffler: Shuffler,
        encrypted_gh_dict: Dict[PYU, HEUObject],
        selected_children_node_indices: List[int],
        all_children_node_indices: List[int],
        children_split_node_selects: List[PYUObject],
        order_map_sub: FedNdarray,
        bucket_num: int,
        bucket_lists: List[PYUObject],
        gradient_encryptor: GradientEncryptor,
        node_num: int,
        node_select_shape: Tuple[int, int],
    ) -> Tuple[PYUObject, PYUObject]:
        bucket_sums_list = [[] for _ in range(self.party_num)]
        bucket_num_plus_one = bucket_num + 1
        if self.params.enable_packbits:
            children_split_node_selects_bits = self.label_holder(packbits_node_selects)(
                children_split_node_selects
            )

        for i, worker in enumerate(self.workers):
            if worker == self.label_holder and self.params.label_holder_feature_only:
                effective_index = i
            if worker != self.label_holder and self.params.label_holder_feature_only:
                continue

            if self.params.enable_packbits:
                children_split_node_selects_worker = worker(unpackbits_node_selects)(
                    children_split_node_selects_bits.to(worker),
                    node_select_shape,
                )
            else:
                children_split_node_selects_worker = children_split_node_selects

            bucket_sums = encrypted_gh_dict[worker].batch_feature_wise_bucket_sum(
                children_split_node_selects_worker,
                order_map_sub.partitions[worker],
                bucket_num_plus_one,
                True,
            )

            self.components.node_wise_cache.batch_collect_node_bucket_sums(
                worker, selected_children_node_indices, bucket_sums
            )
            bucket_sums = self.components.node_wise_cache.batch_get_node_bucket_sum(
                worker, all_children_node_indices
            )
            bucket_sums = [
                bucket_sum[shuffler.create_shuffle_mask(i, j, bucket_lists[i])]
                for j, bucket_sum in zip(all_children_node_indices, bucket_sums)
            ]

            bucket_sums_list[i] = [
                bucket_sum.to(
                    self.label_holder,
                    gradient_encryptor.get_move_config(self.label_holder),
                )
                for bucket_sum in bucket_sums
            ]

        if self.params.label_holder_feature_only:
            bucket_sums_list = [bucket_sums_list[effective_index]]
        level_nodes_G, level_nodes_H = self.label_holder(
            lambda bucket_sums_list, node_num: [
                *zip(
                    *[
                        split_GH(regroup_bucket_sums(bucket_sums_list, idx))
                        for idx in range(node_num)
                    ]
                )
            ],
            num_returns=2,
        )(bucket_sums_list, node_num)
        return level_nodes_G, level_nodes_H

    def remove_pruned_node_cache(self, all_node_indices, gain_is_cost_effective):
        for node_index, gain_effective in zip(all_node_indices, gain_is_cost_effective):
            if not gain_effective:
                self.components.node_wise_cache.reset_node(node_index)

    def reset_cache(self):
        self.components.node_wise_cache.reset()
