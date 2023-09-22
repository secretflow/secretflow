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

import numpy as np

from secretflow.device import PYUObject

from .shuffler_core import Shuffler


class WorkerShuffler:
    def __init__(self, seed: int):
        np.random.seed(seed)
        self.shuffler = Shuffler()

    def reset_shuffle_mask(self):
        self.shuffler.reset_shuffle_mask()

    def reset_shuffle_mask_with_keys(self, keys: List[int]):
        self.shuffler.reset_shuffle_mask_with_keys(keys)

    def create_shuffle_mask(self, key: int, bucket_list: List[PYUObject]) -> List[int]:
        self.shuffler.create_shuffle_mask(key, bucket_list)
        return self.shuffler.get_shuffling_indices(key)

    def is_shuffled(self) -> bool:
        return self.shuffler.is_shuffled()

    def undo_shuffle_mask(self, key: int, index: int) -> int:
        # leave -1 to itself.
        if index == -1:
            return -1
        if self.is_shuffled():
            return self.shuffler.undo_shuffle_mask(key, index)
        else:
            return index

    def undo_shuffle_mask_list_wise(self, split_buckets: List[int]) -> List[int]:
        return [
            self.undo_shuffle_mask(key, index)
            for key, index in enumerate(split_buckets)
        ]

    def undo_shuffle_mask_with_keys(self, split_buckets: List[int], keys) -> List[int]:
        return [
            self.undo_shuffle_mask(key, index)
            for key, index in zip(keys, split_buckets)
        ]
