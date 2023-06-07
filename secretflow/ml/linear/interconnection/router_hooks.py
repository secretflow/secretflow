# Copyright 2023 Ant Group Co., Ltd.
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

from typing import List

import numpy as np
from heu import phe

from secretflow.device import PYUObject, proxy


@proxy(PYUObject)
class WeightArbiter:
    def __init__(self):
        self.kit = phe.setup(phe.SchemaType.ZPaillier, 2048)
        # You can get the public key by following method:
        self.kit.public_key()
        pass

    def sync_with_rs(self, flatten_weights: List[float]):
        return [w * 1.1 for w in flatten_weights]

    def update_weight(self, weights):
        # Input: [10*1_matrix(w0~w9), 10*1_matrix(w10~w19), 10*1_matrix(w20~w29)]

        # each party's weights are n*1 matrix
        # jax.tree_util.tree_flatten cannot flatten 2d-tensor to 1d-list
        weights_flatten = [item[0] for w in weights for item in w]

        # now weights_flatten is a flatten list of length 30: [w0~w29]
        weights_from_rs = self.sync_with_rs(weights_flatten)

        def unflatten():
            idx = 0
            for w in weights:
                yield np.array(weights_from_rs[idx:idx + len(w)]).reshape(-1, 1)
                idx += len(w)

        return list(unflatten())


# 华控隐私路由水平 LR 方案实现
class RouterLrAggrHook:
    def __init__(self, device):
        self.arbiter = device
        self.wa = WeightArbiter(device=device)

    def on_aggregate(self, weights: List[PYUObject]):
        """
        Hook on LR weights aggregation
        Args:
            weights: a list of PYUObject, each PYUObject points to an n*1 matrix, representing n weight values.
                For example, assuming that LR has 30 features and is scattered among 3 participants,
                the input list is: [PYUObject@Alice(w0~w9), PYUObject@Bob(w10~w19), PYUObject@Carol(w20~w29)]

        Returns: The new weights modified by hook in same format of input
        """

        # Send weights to arbiter
        new_weights = self.wa.update_weight(
            [w.to(self.arbiter) for w in weights], _num_returns=len(weights))

        # Move weights to each party
        return [w.to(dev.device) for w, dev in zip(new_weights, weights)]
