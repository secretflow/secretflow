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

from typing import Dict, Iterable, List, Union

import ray

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


@register(DeviceType.SPU)
def to(
    self: SPUObject,
    device: Device,
    spu_vis: str,
    heu_dest_party: str,
    heu_audit_log: str,
):
    if isinstance(device, PYU):
        shares = [
            actor.get_var.remote(self.data) for actor in self.device.actors.values()
        ]
        return device(SPU.outfeed)(self.device.conf, self.data, *shares)
    elif isinstance(device, HEU):
        return spu_to_heu(self, device, heu_dest_party, heu_audit_log)

    raise ValueError(f'Unexpected device type: {type(device)}')


def spu_to_heu(self: SPUObject, heu: HEU, heu_dest_party: str, heu_audit_log: str):
    if heu_dest_party == "auto":
        heu_dest_party = list(heu.evaluator_names())[0]

    if heu_dest_party == heu.sk_keeper_name():
        raise RuntimeError(
            f"move data to heu sk_keeper({heu_dest_party}) is dangerous. If you are "
            f"sure you want to do this, please reveal the data to PYU first, "
            f"then move to HEU sk_keeper."
        )

    heu_parties = list(heu.evaluator_names()) + [heu.sk_keeper_name()]
    assert set(self.device.actors.keys()).issubset(
        heu_parties
    ), f'Mismatch SPU and HEU parties, spu: {list(self.device.actors.keys())}, heu:{heu_parties}'

    # TODO(@xibin.wxb): support pytree
    shards = {
        p: actor.a2h.remote(self.data, heu.cleartext_type)
        for p, actor in self.device.actors.items()
    }
    shards = [
        heu.get_participant(p).encrypt.remote(shard, heu_audit_log)
        if p != heu_dest_party
        else shard
        for p, shard in shards.items()
    ]
    data = heu.get_participant(heu_dest_party).a2h_sum_shards.remote(*shards)

    return HEUObject(heu, data, heu_dest_party)


def _check_psi_protocol(device: SPU, protocol):
    assert protocol in ('kkrt', 'ecdh'), f'unknown psi protocol {protocol}'
    assert (
        len(device.actors) == 2 or len(device.actors) == 3
    ), f'only 2pc or 3pc psi supported'
    if len(device.actors) == 3:
        assert protocol == 'ecdh', f'only ecdh-3pc psi supported'


@register(DeviceType.SPU)
def psi_df(
    device: SPU,
    key: Union[str, Iterable[str]],
    dfs: List['PYUObject'],
    protocol='kkrt',
    sort=True,
) -> List[PYUObject]:
    assert isinstance(device, SPU), f'device must be SPU device'
    assert isinstance(key, (str, Iterable)), f'invalid key, must be str or list of str'
    assert len(set([df.device for df in dfs])) == len(
        dfs
    ), f'dataframe should not be in same PYU device'
    assert len(dfs) == len(
        device.actors
    ), f'unexpected number of dataframes, should be {len(device.actors)}'
    _check_psi_protocol(device, protocol)

    for df in dfs:
        assert isinstance(df, PYUObject), f'dataframe must be in PYU device'
        assert (
            df.device.party in device.actors
        ), f'{df.device} not co-located with {device}'

    res = []
    for df in dfs:
        actor = device.actors[df.device.party]
        res.append(
            PYUObject(df.device, actor.psi_df.remote(key, df.data, protocol, sort))
        )

    return res


@register(DeviceType.SPU)
def psi_csv(
    device: SPU,
    key: Union[str, Iterable[str]],
    input_path: Union[str, Dict[Device, str]],
    output_path: Union[str, Dict[Device, str]],
    protocol='kkrt',
    sort=True,
):
    assert isinstance(device, SPU), f'device must be SPU device'
    assert isinstance(key, (str, Iterable)), f'invalid key, must be str or list of str'
    assert isinstance(input_path, (str, Dict)), f'input_path must be str or dict of str'
    assert isinstance(
        output_path, (str, Dict)
    ), f'input_path must be str or dict of str'
    _check_psi_protocol(device, protocol)

    if isinstance(input_path, Dict):
        assert (
            input_path.keys() == output_path.keys()
        ), f'mismatch input_path and out_path devices'
        assert len(input_path) == len(
            device.actors
        ), f'unexpected number of dataframes, should be {len(device.actors)}'

        for dev in input_path.keys():
            assert (
                dev.party in device.actors
            ), f'input_path {dev} not co-located with {device}'

        for dev in output_path.keys():
            assert (
                dev.party in device.actors
            ), f'output_path {dev} not co-located with {device}'

    res = []
    if isinstance(input_path, str):
        assert isinstance(
            output_path, str
        ), f'input_path and output_path must be same types'
        for actor in device.actors:
            res.append(
                actor.psi_csv.remote(key, input_path, output_path, protocol, sort)
            )
    else:
        for dev, ipath in input_path.items():
            opath = output_path[dev]
            actor = device.actors[dev.party]
            res.append(actor.psi_csv.remote(key, ipath, opath, protocol, sort))

    # wait for all tasks done
    return ray.get(res)
