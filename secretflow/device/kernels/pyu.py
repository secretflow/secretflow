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

from secretflow.device import (HEU, PYU, SPU, SPUIO, Device, DeviceType,
                               HEUObject, PYUObject, SPUObject, register)


@register(DeviceType.PYU)
def to(self: PYUObject, device: Device, config):
    assert isinstance(device, Device), f'Expect a device but got {type(device)}'

    if isinstance(device, PYU):
        return PYUObject(device, self.data)
    elif isinstance(device, SPU):
        assert (
            config.spu_vis == 'secret' or config.spu_vis == 'public'
        ), f'vis must be public or secret'

        vtype = (
            Visibility.VIS_PUBLIC
            if config.spu_vis == 'public'
            else Visibility.VIS_SECRET
        )

        def run_spu_io(data, runtime_config, world_size, vtype):
            io = SPUIO(runtime_config, world_size)
            return io.make_shares(data, vtype)

        meta, *shares = self.device(run_spu_io, num_returns=(1 + device.world_size))(
            self.data, device.conf, device.world_size, vtype
        )
        return SPUObject(
            device, meta.data, device.infeed_shares([share.data for share in shares])
        )

    elif isinstance(device, HEU):  # PYU -> HEU, pure local operation
        if config.heu_dest_party == 'auto':
            config.heu_dest_party = list(device.evaluator_names())[0]

        data = device.get_participant(self.device.party).encode.remote(
            self.data, config.heu_encoder
        )
        return HEUObject(device, data, self.device.party, True).to(device, config)

    raise ValueError(f'Unexpected device type: {type(device)}')
