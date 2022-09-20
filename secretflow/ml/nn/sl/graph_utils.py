# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import dgl


class NodeDataLoader:
    def __init__(
        self,
        g,
        indices,
        sampler,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        seed=None,
    ) -> None:
        assert not shuffle or seed is not None, "seed must be set if shuffle"
        self.g = g
        self.indices = indices
        self.collator = dgl.dataloading.NodeCollator(g, indices, sampler)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._steps_per_epoch = (
            len(indices) // batch_size
            if drop_last
            else (len(indices) + batch_size - 1) // batch_size
        )
        self._random_state = np.random.RandomState(seed=seed)

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch

    def __iter__(self):
        if self.shuffle:
            self.indices = self._random_state.permutation(self.indices)
        self._step = -1
        return self

    def __next__(self):
        self._step += 1
        # all data is less than one batch
        if self._step == self._steps_per_epoch:
            raise StopIteration

        beg = self._step * self.batch_size
        end = (
            beg + self.batch_size
            if (beg + self.batch_size) < len(self.indices)
            else len(self.indices)
        )
        batch_indices = self.indices[beg:end]
        input_nodes, output_nodes, blocks = self.collator.collate(batch_indices)

        # NOTE: NodeDataLoader never raise StopIteration, since `SLModel` controls
        # the number of epochs and steps per epoch.
        if self._step == self._steps_per_epoch - 1:
            if self.shuffle:
                self.indices = self._random_state.permutation(self.indices)
            self._step = -1

        return self.g, input_nodes, output_nodes, blocks
