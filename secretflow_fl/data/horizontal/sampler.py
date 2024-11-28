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
import tensorflow as tf


class PoissonDataSampler(tf.keras.utils.Sequence):
    "Generates data with poisson sampling"

    def __init__(self, x, y, s_w, sampling_rate, **kwargs):
        "Initialization"
        self.x = x
        self.y = y
        self.s_w = s_w
        self.sampling_rate = sampling_rate
        self.num_examples = len(self.y)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        "Generate one batch of data"
        while True:
            sample_size = np.random.binomial(self.num_examples, self.sampling_rate)
            if sample_size > 0:
                break
        indices = np.random.choice(self.num_examples, sample_size, replace=False)
        if self.s_w is None:
            return self.x[indices], self.y[indices]
        else:
            return self.x[indices], self.y[indices], self.s_w[indices]

    def set_random_seed(self, random_seed):
        np.random.seed(random_seed)
