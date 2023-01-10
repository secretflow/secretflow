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

from typing import Dict, List, Union

import secretflow.distributed as sfd
from secretflow.device import (
    HEU,
    PYU,
    SPU,
    SPUIO,
    Device,
    DeviceType,
    HEUObject,
    PYUObject,
    SPUObject,
    register,
)


@register(DeviceType.SPU)
def to(self: SPUObject, device: Device, config):
    if isinstance(device, PYU):

        def reveal(conf, world_size, refs, meta):
            io = SPUIO(conf, world_size)
            return io.reconstruct(refs, meta)

        return device(reveal)(
            self.device.conf,
            self.device.world_size,
            self.device.outfeed_shares(self.shares_name),
            self.meta,
        )

    if isinstance(device, SPU):
        # same spu
        if self.device == device:
            return self

        # send to another spu.
        assert (
            device.conf.protocol == self.device.conf.protocol
            and device.conf.field == self.device.conf.field
            and device.conf.fxp_fraction_bits == self.device.conf.fxp_fraction_bits
            and device.world_size == self.device.world_size
        )

        shares = self.device.outfeed_shares(self.shares_name)
        shares_name = device.infeed_shares(shares)

        # TODO: do we need reshare shares.
        return SPUObject(device, self.meta, shares_name)

    if isinstance(device, HEU):
        return spu_to_heu(self, device, config)

    raise ValueError(f'Unexpected device type: {type(device)}')


def spu_to_heu(self: SPUObject, heu: HEU, config):
    if config.heu_dest_party == "auto":
        config.heu_dest_party = list(heu.evaluator_names())[0]

    if config.heu_dest_party == heu.sk_keeper_name():
        raise RuntimeError(
            f"move data to heu sk_keeper({config.heu_dest_party}) is dangerous. If you are "
            f"sure you want to do this, please reveal the data to PYU first, "
            f"then move to HEU sk_keeper."
        )

    heu_parties = list(heu.evaluator_names()) + [heu.sk_keeper_name()]
    assert set(self.device.actors.keys()).issubset(
        heu_parties
    ), f'Mismatch SPU and HEU parties, spu: {list(self.device.actors.keys())}, heu:{heu_parties}'

    # TODO(@xibin.wxb): support pytree
    shards = {
        p: actor.a2h.remote(ref, heu.cleartext_type, heu.schema)
        for (p, actor), ref in zip(
            self.device.actors.items(), self.device.outfeed_shares(self.shares_name)
        )
    }
    shards = [
        heu.get_participant(p).encrypt.remote(shard, config.heu_audit_log)
        if p != config.heu_dest_party
        else shard
        for p, shard in shards.items()
    ]
    data = heu.get_participant(config.heu_dest_party).a2h_sum_shards.remote(*shards)

    return HEUObject(heu, data, config.heu_dest_party)


@register(DeviceType.SPU)
def psi_df(
    device: SPU,
    key: Union[str, List[str], Dict[Device, List[str]]],
    dfs: List['PYUObject'],
    receiver: str,
    protocol='KKRT_PSI_2PC',
    precheck_input=True,
    sort=True,
    broadcast_result=True,
    bucket_size=1 << 20,
    curve_type="CURVE_25519",
    preprocess_path=None,
    ecdh_secret_key_path=None,
) -> List[PYUObject]:
    assert isinstance(device, SPU), f'device must be SPU device'
    assert isinstance(
        key, (str, List, Dict)
    ), f'invalid key, must be str of list of str or dict of str list'
    assert len(set([df.device for df in dfs])) == len(
        dfs
    ), f'dataframe should not be in same PYU device'
    assert len(dfs) == len(
        device.actors
    ), f'unexpected number of dataframes, should be {len(device.actors)}'

    for df in dfs:
        assert isinstance(df, PYUObject), f'dataframe must be in PYU device'
        assert (
            df.device.party in device.actors
        ), f'{df.device} not co-located with {device}'

    res = []
    for df in dfs:
        actor = device.actors[df.device.party]
        k = key[df.device] if isinstance(key, Dict) else key
        res.append(
            PYUObject(
                df.device,
                actor.psi_df.remote(
                    k,
                    df.data,
                    receiver,
                    protocol,
                    precheck_input,
                    sort,
                    broadcast_result,
                    bucket_size,
                    curve_type,
                    preprocess_path,
                    ecdh_secret_key_path,
                ),
            )
        )

    return res


@register(DeviceType.SPU)
def psi_csv(
    device: SPU,
    key: Union[str, List[str], Dict[Device, List[str]]],
    input_path: Union[str, Dict[Device, str]],
    output_path: Union[str, Dict[Device, str]],
    receiver: str,
    protocol='KKRT_PSI_2PC',
    precheck_input=True,
    sort=True,
    broadcast_result=True,
    bucket_size=1 << 20,
    curve_type="CURVE_25519",
    preprocess_path=None,
    ecdh_secret_key_path=None,
):
    assert isinstance(device, SPU), f'device must be SPU device'
    assert isinstance(
        key, (str, List, Dict)
    ), f'invalid key, must be str of list of str or dict of list str'
    assert isinstance(input_path, (str, Dict)), f'input_path must be str or dict of str'
    assert isinstance(
        output_path, (str, Dict)
    ), f'output_path must be str or dict of str'

    if isinstance(key, Dict):
        if isinstance(input_path, Dict):
            assert (
                key.keys() == input_path.keys() == output_path.keys()
            ), f'mismatch key & input_path & out_path devices'
            for dev in key.keys():
                assert (
                    dev.party in device.actors
                ), f'key {dev} not co-located with {device}'

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
            k = key[actor] if isinstance(key, Dict) else key
            res.append(
                actor.psi_csv.remote(
                    k,
                    input_path,
                    output_path,
                    receiver,
                    protocol,
                    precheck_input,
                    sort,
                    broadcast_result,
                    bucket_size,
                    curve_type,
                    preprocess_path,
                    ecdh_secret_key_path,
                )
            )
    else:
        for dev, ipath in input_path.items():
            opath = output_path[dev]
            actor = device.actors[dev.party]
            k = key[dev] if isinstance(key, Dict) else key
            res.append(
                actor.psi_csv.remote(
                    k,
                    ipath,
                    opath,
                    receiver,
                    protocol,
                    precheck_input,
                    sort,
                    broadcast_result,
                    bucket_size,
                    curve_type,
                    preprocess_path,
                    ecdh_secret_key_path,
                )
            )

    # wait for all tasks done
    return sfd.get(res)


@register(DeviceType.SPU)
def psi_join_df(
    device: SPU,
    key: Union[str, List[str], Dict[Device, List[str]]],
    dfs: List['PYUObject'],
    receiver: str,
    join_party: str,
    protocol='KKRT_PSI_2PC',
    precheck_input=True,
    bucket_size=1 << 20,
    curve_type="CURVE_25519",
) -> List[PYUObject]:
    assert isinstance(device, SPU), f'device must be SPU device'
    assert isinstance(
        key, (str, List, Dict)
    ), f'invalid key, must be str of list of str or dict of str list'
    assert len(set([df.device for df in dfs])) == len(
        dfs
    ), f'dataframe should not be in same PYU device'
    assert len(dfs) == len(
        device.actors
    ), f'unexpected number of dataframes, should be {len(device.actors)}'

    for df in dfs:
        assert isinstance(df, PYUObject), f'dataframe must be in PYU device'
        assert (
            df.device.party in device.actors
        ), f'{df.device} not co-located with {device}'

    res = []
    for df in dfs:
        actor = device.actors[df.device.party]
        k = key[df.device] if isinstance(key, Dict) else key
        res.append(
            PYUObject(
                df.device,
                actor.psi_join_df.remote(
                    k,
                    df.data,
                    receiver,
                    join_party,
                    protocol,
                    precheck_input,
                    bucket_size,
                    curve_type,
                ),
            )
        )

    return res


@register(DeviceType.SPU)
def psi_join_csv(
    device: SPU,
    key: Union[str, List[str], Dict[Device, List[str]]],
    input_path: Union[str, Dict[Device, str]],
    output_path: Union[str, Dict[Device, str]],
    receiver: str,
    join_party: str,
    protocol='KKRT_PSI_2PC',
    precheck_input=True,
    bucket_size=1 << 20,
    curve_type="CURVE_25519",
):
    assert isinstance(device, SPU), f'device must be SPU device'
    assert isinstance(
        key, (str, List, Dict)
    ), f'invalid key, must be str of list of str or dict of list str'
    assert isinstance(input_path, (str, Dict)), f'input_path must be str or dict of str'
    assert isinstance(
        output_path, (str, Dict)
    ), f'output_path must be str or dict of str'

    if isinstance(key, Dict):
        if isinstance(input_path, Dict):
            assert (
                key.keys() == input_path.keys() == output_path.keys()
            ), f'mismatch key & input_path & out_path devices'
            for dev in key.keys():
                assert (
                    dev.party in device.actors
                ), f'key {dev} not co-located with {device}'

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
            k = key[actor] if isinstance(key, Dict) else key
            res.append(
                actor.psi_join_csv.remote(
                    k,
                    input_path,
                    output_path,
                    receiver,
                    join_party,
                    protocol,
                    precheck_input,
                    bucket_size,
                    curve_type,
                )
            )
    else:
        for dev, ipath in input_path.items():
            opath = output_path[dev]
            actor = device.actors[dev.party]
            k = key[dev] if isinstance(key, Dict) else key
            res.append(
                actor.psi_join_csv.remote(
                    k,
                    ipath,
                    opath,
                    receiver,
                    join_party,
                    protocol,
                    precheck_input,
                    bucket_size,
                    curve_type,
                )
            )

    # wait for all tasks done
    return sfd.get(res)
