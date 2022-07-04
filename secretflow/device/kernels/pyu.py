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

from spu import Visibility

from secretflow.device import (
    HEU,
    SPU,
    PYU,
    Device,
    DeviceType,
    HEUObject,
    SPUObject,
    PYUObject,
    register,
)


@register(DeviceType.PYU)
def to(
    self: PYUObject,
    device: Device,
    spu_vis: str,
    heu_dest_party: str,
    heu_audit_log: str,
):
    assert isinstance(device, Device), f'Expect a device but got {type(device)}'

    if isinstance(device, PYU):
        return PYUObject(device, self.data)
    elif isinstance(device, SPU):
        assert (
            spu_vis == 'secret' or spu_vis == 'public'
        ), f'vis must be public or secret'
        spu_vis = (
            Visibility.VIS_PUBLIC if spu_vis == 'public' else Visibility.VIS_SECRET
        )

        value_shares = self.device(SPU.infeed, num_returns=len(device.actors) + 1)(
            device.cluster_def, str(self.data), self.data, spu_vis
        )
        shares, tree = value_shares[:-1], value_shares[-1]

        for i, actor in enumerate(device.actors.values()):
            actor.set_var.remote(shares[i].data)

        return SPUObject(device, tree.data)
    elif isinstance(device, HEU):  # PYU -> HEU, pure local operation
        if heu_dest_party == 'auto':
            heu_dest_party = list(device.evaluator_names())[0]
        return HEUObject(device, self.data, self.device.party, True).to(
            device, heu_dest_party=heu_dest_party, heu_audit_log=heu_audit_log
        )

    raise ValueError(f'Unexpected device type: {type(device)}')
