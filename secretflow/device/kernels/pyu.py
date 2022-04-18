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

import ray
from ppu.binding import Visibility

from secretflow.device.device import PPU, PPUObject, HEUObject, HEU, PYU, PYUObject
from secretflow.device.device import register, DeviceType, Device


@register(DeviceType.PYU)
def to(self, device: Device, vis):
    assert isinstance(device, Device), f'Expect a device but got {type(device)}'

    if isinstance(device, PYU):
        return PYUObject(device, self.data)
    elif isinstance(device, PPU):
        assert vis == 'secret' or vis == 'public', f'vis must be public or secret'
        vis = Visibility.VIS_PUBLIC if vis == 'public' else Visibility.VIS_SECRET

        obj = self.device(PPU.infeed)(device.cluster_def, str(id(self)), self.data, vis)
        shares, data = ray.get(obj.data)

        # NOTE: batch set_var to avoid too fine-grained tasks
        for i, actor in enumerate(device.actors.values()):
            names, vars = [], []
            for name, share in shares.items():
                names.append(name)
                vars.append(share[i])
            actor.set_var.remote(names, *vars)

        return PPUObject(device, data)
    elif isinstance(device, HEU):
        if self.device.party == device.evaluator:  # strong trust
            return HEUObject(device, self.data)
        else:  # weak trust
            data = device.encrypt.options(resources={self.device.party: 1}).remote(device.pk, self.data)
            return HEUObject(device, data)

    raise ValueError(f'Unexpected device type: {type(device)}')
