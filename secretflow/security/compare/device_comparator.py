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

from secretflow.device.device import PPU, PYU, DeviceObject
from secretflow.device.device.base import reveal
from secretflow.security.compare.comparator import Comparator


@dataclass
class DeviceComparator(Comparator):
    device: Union[PYU, PPU]
    
    #TODO: reveal由用户来决定。
    @reveal
    def min(self, data: List[DeviceObject], axis=None):
        assert data, f'Data to get min can not be None or empty'
        data = [d.to(self.device) for d in data]

        def _min(data, axis):
            return jnp.min(jnp.array(data), axis=axis)

        return self.device(_min, static_argnames='axis')(data, axis=axis)

    @reveal
    def max(self, data: List[DeviceObject], axis=None):
        assert data, f'Data to get max can not be None or empty'
        data = [d.to(self.device) for d in data]

        def _max(data, axis):
            return jnp.max(jnp.array(data), axis=axis)

        return self.device(_max, static_argnames='axis')(data, axis=axis)
