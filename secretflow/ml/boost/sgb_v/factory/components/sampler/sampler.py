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
from typing import List, Tuple, Union

import numpy as np

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device import PYUObject
from secretflow.ml.boost.sgb_v.core.params import default_params
from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

from ..component import Component, Devices, print_params
from .sample_actor import SampleActor


@dataclass
class SamplerParams:
    """
    'rowsample_by_tree': float. Row sub sample ratio of the training instances.
        default: 1
        range: (0, 1]
    'colsample_by_tree': float. Col sub sample ratio of columns when constructing each tree.
        default: 1
        range: (0, 1]
    'seed': int. Pseudorandom number generator seed.
        default: 1212
    'label_holder_feature_only': bool. affects col sampling.
        default: False
        if turned on, all non-label holder's col sample rate will be 0.
    'enable_goss': bool. whether enable GOSS, see lightGBM's paper for more understanding in GOSS.
        default: False
    'top_rate': float. GOSS-specific parameter. The fraction of large gradients to sample.
        default: 0.3
        range: (0, 1), but top_rate + bottom_rate < 1
    'bottom_rate': float. GOSS-specific parameter. The fraction of small gradients to sample.
        default: 0.5
        range: (0, 1), but top_rate + bottom_rate < 1
    """

    rowsample_by_tree: float = default_params.rowsample_by_tree
    colsample_by_tree: float = default_params.colsample_by_tree
    seed: int = default_params.seed
    label_holder_feature_only: bool = False
    enable_goss: bool = default_params.enable_goss
    top_rate: float = default_params.top_rate
    bottom_rate: float = default_params.bottom_rate


class Sampler(Component):
    def __init__(self):
        self.params = SamplerParams()

    def show_params(self):
        print_params(self.params)

    def set_params(self, params: dict):
        subsample = params.get('rowsample_by_tree', default_params.rowsample_by_tree)
        colsample = params.get('colsample_by_tree', default_params.colsample_by_tree)
        top_rate = params.get('top_rate', default_params.top_rate)
        bottom_rate = params.get('bottom_rate', default_params.bottom_rate)

        assert (
            bottom_rate + top_rate < 1
        ), f"the sum of top_rate and bottom_rate should be less than 1, got {bottom_rate + top_rate}"

        self.params.rowsample_by_tree = subsample
        self.params.colsample_by_tree = colsample
        self.params.seed = params.get('seed', default_params.seed)
        self.params.label_holder_feature_only = params.get(
            'label_holder_feature_only', False
        )

        self.params.enable_goss = params.get('enable_goss', default_params.enable_goss)
        self.params.top_rate = top_rate
        self.params.bottom_rate = bottom_rate

    def get_params(self, params: dict):
        params['seed'] = self.params.seed
        params['rowsample_by_tree'] = self.params.rowsample_by_tree
        params['colsample_by_tree'] = self.params.colsample_by_tree
        params['label_holder_feature_only'] = self.params.label_holder_feature_only
        params['enable_goss'] = self.params.enable_goss
        params['top_rate'] = self.params.top_rate
        params['bottom_rate'] = self.params.bottom_rate

    def set_devices(self, devices: Devices):
        self.label_holder = devices.label_holder
        self.workers = devices.workers

    def set_actors(self, actors: SGBActor):
        self.sample_actors = {actor.device: actor for actor in actors}
        for actor in self.sample_actors.values():
            actor.register_class('SampleActor', SampleActor, self.params.seed)

    def del_actors(self):
        del self.sample_actors

    def generate_col_choices(
        self, feature_buckets: List[PYUObject]
    ) -> Tuple[List[PYUObject], List[PYUObject]]:
        """Generate column sample choices.

        Args:
            feature_buckets (List[PYUObject]): Behind PYUObject is List[int], bucket num for each feature.

        Returns:
            Tuple[List[PYUObject], List[PYUObject]]: first list is column choices, second is total number of buckets after sampling
        """
        colsample = self.params.colsample_by_tree

        if self.params.label_holder_feature_only:
            col_choices, total_buckets = zip(
                *[
                    self.sample_actors[fb.device].invoke_class_method_two_ret(
                        'SampleActor',
                        'generate_one_partition_col_choices',
                        colsample,
                        fb,
                    )
                    if fb.device == self.label_holder
                    else self.sample_actors[fb.device].invoke_class_method_two_ret(
                        'SampleActor', 'generate_one_partition_col_choices', 0, fb
                    )
                    for fb in feature_buckets
                ]
            )
        else:
            col_choices, total_buckets = zip(
                *[
                    self.sample_actors[fb.device].invoke_class_method_two_ret(
                        'SampleActor',
                        'generate_one_partition_col_choices',
                        colsample,
                        fb,
                    )
                    for fb in feature_buckets
                ]
            )
        return col_choices, total_buckets

    def generate_row_choices(
        self, row_num: Union[PYUObject, int], g: PYUObject
    ) -> Tuple[Union[None, np.ndarray], Union[None, np.ndarray]]:
        """Sample rows,
        either in a goss style or normal style based on config

        Args:
            row_num (int): row number
            g (PYUObject): gradient

        Returns:
            Tuple[Union[None, np.ndarray], Union[None, np.ndarray]]:
                1. row choices
                2. weight (for info gain), None if not GOSS-enabled
        """
        if self.params.enable_goss:
            top_rate = self.params.top_rate
            bottom_rate = self.params.bottom_rate
            return self.sample_actors[g.device].invoke_class_method_two_ret(
                'SampleActor', 'goss', row_num, g, top_rate, bottom_rate
            )
        else:
            sample_rate = self.params.rowsample_by_tree
            choices = self.sample_actors[g.device].invoke_class_method(
                'SampleActor', 'generate_row_choices', row_num, sample_rate
            )
            return choices, None

    def should_row_subsampling(self) -> bool:
        return self.params.rowsample_by_tree < 1 or self.params.enable_goss

    def _apply_vector_sampling(
        self,
        x: PYUObject,
        indices: Union[PYUObject, np.ndarray],
    ):
        """Sample x for a single partition. Assuming we have a column vector.
        Assume the indices was generated from row sampling by sampler"""
        if self.params.rowsample_by_tree < 1:
            return x.device(lambda x, indices: x.reshape(-1, 1)[indices, :])(x, indices)
        else:
            return x.device(lambda x: x.reshape(-1, 1))(x)

    def apply_vector_sampling_weighted(
        self,
        x: PYUObject,
        indices: Union[PYUObject, np.ndarray],
        weight: Union[PYUObject, None] = None,
    ):
        if self.params.enable_goss:
            return x.device(
                lambda x, indices, weight: (
                    np.multiply(x.reshape(-1)[indices], weight.reshape(-1))
                ).reshape(-1, 1)
            )(
                x,
                indices,
                weight,
            )
        else:
            return self._apply_vector_sampling(x, indices)

    def apply_v_fed_sampling(
        self,
        X: FedNdarray,
        row_choices: Union[None, np.ndarray, PYUObject] = None,
        col_choices: List[Union[None, np.ndarray, PYUObject]] = [],
    ) -> FedNdarray:
        """Sample X based on row choices and col choices.
        Assume the choices were generated by sampler.

        Args:
            X (FedNdarray): Array to sample from
            row_choices (Union[None, np.ndarray, PYUObject]): row sampling choices. devices are assumed to be ordered as X.
            col_choices (List[Union[None, np.ndarray,PYUObject]): col sampling choices. devices are assumed to be ordered as X.

        Returns:
            X_sub (FedNdarray): subsampled X
            shape (Tuple[int, int]): shape of X_sub
        """
        X_sub = X
        # sample cols and rows of bucket_map
        if self.params.colsample_by_tree < 1 and self.should_row_subsampling():
            # sub choices is stored in context owned by label_holder and shared to all workers.
            X_sub = FedNdarray(
                partitions={
                    pyu: pyu(lambda x, y, z: x[y, :][:, z])(
                        partition,
                        row_choices.to(pyu)
                        if isinstance(row_choices, PYUObject)
                        else row_choices,
                        col_choices[i],
                    )
                    for i, (pyu, partition) in enumerate(X.partitions.items())
                },
                partition_way=PartitionWay.VERTICAL,
            )
        # only sample cols
        elif self.params.colsample_by_tree < 1:
            X_sub = FedNdarray(
                partitions={
                    pyu: pyu(lambda x, y: x[:, y])(partition, col_choices[i])
                    for i, (pyu, partition) in enumerate(X.partitions.items())
                },
                partition_way=PartitionWay.VERTICAL,
            )
        # only sample rows
        elif self.should_row_subsampling():
            X_sub = FedNdarray(
                partitions={
                    pyu: pyu(lambda x, y: x[y, :])(
                        partition,
                        row_choices.to(pyu)
                        if isinstance(row_choices, PYUObject)
                        else row_choices,
                    )
                    for pyu, partition in X.partitions.items()
                },
                partition_way=PartitionWay.VERTICAL,
            )
        return X_sub
