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

from dataclasses import dataclass
from typing import List, Union

import jax.numpy as jnp

from secretflow.device.device import PYU, PPU, DeviceObject, reveal
from secretflow.security.aggregation.aggregator import Aggregator


@dataclass
class DeviceAggregator(Aggregator):
    device: Union[PYU, PPU]

    # TODO(@zhouaihui): 此处是否reveal需要交给用户来选择
    @reveal
    def sum(self, data: List[DeviceObject], axis=None):
        assert data, 'Data to aggregate should not be None or empty!'
        data = [d.to(self.device) for d in data]

        def _sum(data, axis):
            if isinstance(data[0], (list, tuple)):
                return [jnp.sum(jnp.array(element), axis=axis) for element in zip(*data)]
            else:
                return jnp.sum(jnp.array(data), axis=axis)

        return self.device(_sum, static_argnames='axis')(data, axis=axis)

    # TODO(@zhouaihui): 此处是否reveal需要交给用户来选择
    @reveal
    def average(self, data: List[DeviceObject], axis=None, weights=None):
        assert data, 'Data to aggregate should not be None or empty!'
        data = [d.to(self.device) for d in data]
        if isinstance(weights, (list, tuple)):
            weights = [w.to(self.device) if isinstance(w, DeviceObject) else w for w in weights]

        def _average(data, axis, weights):
            if isinstance(data[0], (list, tuple)):
                return [jnp.average(element, axis=axis, weights=weights) for element in zip(*data)]
            else:
                return jnp.average(data, axis=axis, weights=weights)

        return self.device(_average, static_argnames='axis')(data, axis=axis, weights=weights)
