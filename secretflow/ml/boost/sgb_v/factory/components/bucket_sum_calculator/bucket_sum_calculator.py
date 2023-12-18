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
from secretflow.device import HEUObject, PYU, PYUObject
from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

from ....core.pure_numpy_ops.bucket_sum import batch_select_sum, regroup_bucket_sums
from ....core.pure_numpy_ops.grad import split_GH
from ....core.pure_numpy_ops.node_select import (
    packbits_node_selects,
    unpackbits_node_selects,
)
from ..cache.level_wise_cache import LevelWiseCache
from ..component import Composite, Devices, print_params
from ..gradient_encryptor import GradientEncryptor
from ..logging import LoggingParams, LoggingTools
from ..shuffler import Shuffler


@dataclass
class BucketSumCalculatorParams:
    """
    'label_holder_feature_only': bool. if true, non-label holder will not do bucket sum
        default: False
    'enable_packbits': bool. if true, turn on packbits transmission.
        default: False
    """

    label_holder_feature_only: bool = False
    enable_packbits: bool = False


@dataclass
class BucketSumCalculatorComponents:
    level_wise_cache: LevelWiseCache = LevelWiseCache()


class BucketSumCalculator(Composite):
    def __init__(self):
        self.components = BucketSumCalculatorComponents()
        self.logging_params = LoggingParams()
        self.params = BucketSumCalculatorParams()

    def show_params(self):
        print_params(self.logging_params)
        print_params(self.params)

    def set_params(self, params: dict):
        self.logging_params = LoggingTools.logging_params_from_dict(params)
        self.params.label_holder_feature_only = bool(
            params.get('label_holder_feature_only', False)
        )
        self.params.enable_packbits = bool(params.get('enable_packbits', False))

    def get_params(self, params: dict):
        LoggingTools.logging_params_write_dict(params, self.logging_params)
        params['label_holder_feature_only'] = self.params.label_holder_feature_only
        params['enable_packbits'] = self.params.enable_packbits

    def set_devices(self, devices: Devices):
        super().set_devices(devices)
        self.label_holder = devices.label_holder
        self.workers = devices.workers
        self.party_num = len(self.workers)

    def set_actors(self, actors: List[SGBActor]):
        super().set_actors(actors)

    def del_actors(self):
        super().del_actors()

    @LoggingTools.enable_logging
    def calculate_bucket_sum_level_wise(
        self,
        shuffler: Shuffler,
        encrypted_gh_dict: Dict[PYU, HEUObject],
        children_split_node_selects: PYUObject,  # inner type is List[np.array]
        is_lefts: List[bool],
        order_map_sub: FedNdarray,
        bucket_num: int,
        bucket_lists: List[PYUObject],
        gradient_encryptor: GradientEncryptor,
        node_num: int,
        node_select_shape: Tuple[int, int],
    ) -> Tuple[PYUObject, PYUObject]:
        bucket_sums_list = [[] for _ in range(self.party_num)]
        bucket_num_plus_one = bucket_num + 1
        shuffler.reset_shuffle_masks()
        self.components.level_wise_cache.reset_level_caches()
        enable = self.params.enable_packbits
        if enable:
            children_split_node_selects_bits = self.label_holder(packbits_node_selects)(
                children_split_node_selects
            )
        for i, worker in enumerate(self.workers):
            if worker != self.label_holder:
                if self.params.label_holder_feature_only:
                    continue
                else:
                    if enable:
                        children_split_node_selects_worker = worker(
                            unpackbits_node_selects
                        )(
                            children_split_node_selects_bits.to(worker),
                            node_select_shape,
                        )
                    else:
                        children_split_node_selects_worker = children_split_node_selects
                    bucket_sums = encrypted_gh_dict[
                        worker
                    ].batch_feature_wise_bucket_sum(
                        children_split_node_selects_worker,
                        order_map_sub.partitions[worker],
                        bucket_num_plus_one,
                        True,
                    )
                    self.components.level_wise_cache.collect_level_node_GH(
                        worker, bucket_sums, is_lefts
                    )
                    bucket_sums = self.components.level_wise_cache.get_level_nodes_GH(
                        worker
                    )
                    bucket_sums = [
                        bucket_sum[shuffler.create_shuffle_mask(i, j, bucket_lists[i])]
                        for j, bucket_sum in enumerate(bucket_sums)
                    ]

                    bucket_sums_list[i] = [
                        bucket_sum.to(
                            self.label_holder,
                            gradient_encryptor.get_move_config(self.label_holder),
                        )
                        for bucket_sum in bucket_sums
                    ]
            else:
                bucket_sums = self.label_holder(batch_select_sum)(
                    encrypted_gh_dict[worker],
                    children_split_node_selects,
                    order_map_sub.partitions[worker],
                    bucket_num_plus_one,
                )

                self.components.level_wise_cache.collect_level_node_GH(
                    worker, bucket_sums, is_lefts
                )
                bucket_sums = self.components.level_wise_cache.get_level_nodes_GH(
                    worker
                )
                bucket_sums_list[i] = bucket_sums
                if self.params.label_holder_feature_only:
                    effective_index = i
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

    def update_level_cache(self, is_last_level, gain_is_cost_effective):
        self.components.level_wise_cache.update_level_cache(
            is_last_level, gain_is_cost_effective
        )

    def reset_cache(self):
        self.components.level_wise_cache.reset()
