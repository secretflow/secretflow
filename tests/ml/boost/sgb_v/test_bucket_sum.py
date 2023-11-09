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

import os

import numpy as np

from secretflow.device.driver import reveal
from secretflow.ml.boost.sgb_v.core.pure_numpy_ops.bucket_sum import (
    batch_select_sum,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_bucket_sum(sf_production_setup_devices_aby3):
    alice = sf_production_setup_devices_aby3.alice
    sample_num = 80 * 100
    feature_num = 300
    bucket_num = 100
    node_num = 2**4
    gh = np.random.random((sample_num, 2))
    children_nodes_selects = np.random.randint(0, node_num, (1, sample_num))
    children_nodes_selects = [
        (children_nodes_selects == i).astype(int) for i in range(node_num)
    ]
    order_map = np.random.randint(0, bucket_num, (sample_num, feature_num))

    result = reveal(
        alice(batch_select_sum)(gh, children_nodes_selects, order_map, bucket_num)
    )
    assert len(result) == node_num
