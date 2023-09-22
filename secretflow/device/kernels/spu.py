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

from typing import Callable, Dict, List, Union

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
from secretflow.utils.progress import ProgressData
from secretflow.device.device.base import register_to
from secretflow.device.device.heu import HEUMoveConfig


@register_to(DeviceType.SPU, DeviceType.PYU)
def spu_to_pyu(self: SPUObject, pyu: Device, config: HEUMoveConfig = None):
    assert isinstance(pyu, PYU), f'Expect a PYU but got {type(pyu)}.'
    if config is None:
        config = HEUMoveConfig()

    def reveal(conf, world_size, io_info, share_chunks, meta):
        io = SPUIO(conf, world_size)
        return io.reconstruct(share_chunks, io_info, meta)

    return pyu(reveal)(
        self.device.conf,
        self.device.world_size,
        *self.device.outfeed_shares(self.shares_name),
        self.meta,
    )


@register_to(DeviceType.SPU, DeviceType.SPU)
def spu_to_spu(self: SPUObject, spu: SPU):
    assert isinstance(spu, SPU), f'Expect an SPU but got {type(spu)}.'
    # same spu
    if self.device == spu:
        return self

    # send to another spu.
    assert (
        spu.conf.protocol == self.device.conf.protocol
        and spu.conf.field == self.device.conf.field
        and spu.conf.fxp_fraction_bits == self.device.conf.fxp_fraction_bits
        and spu.world_size == self.device.world_size
    )

    io_info, shares_chunk = self.device.outfeed_shares(self.shares_name)
    shares_name = spu.infeed_shares(io_info, shares_chunk)

    # TODO: do we need reshare shares.
    return SPUObject(spu, self.meta, shares_name)


@register_to(DeviceType.SPU, DeviceType.HEU)
def spu_to_heu(self: SPUObject, heu: Device, config: HEUMoveConfig = None):
    assert isinstance(heu, HEU), f'Expect an HEU but got {type(heu)}.'
    if config is None:
        config = HEUMoveConfig()

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
    io_info, shares_chunk = self.device.outfeed_shares(self.shares_name)
    assert (
        len(shares_chunk) % len(self.device.actors) == 0
    ), f"{len(shares_chunk)} % {len(self.device.actors)}"
    chunks_count_pre_party = int(len(shares_chunk) / len(self.device.actors))
    chunks_pre_party = [
        shares_chunk[i * chunks_count_pre_party : (i + 1) * chunks_count_pre_party]
        for i in range(len(self.device.actors))
    ]
    shards = {
        p: actor.a2h.remote(io_info, heu.cleartext_type, heu.schema, *chunks)
        for (p, actor), chunks in zip(self.device.actors.items(), chunks_pre_party)
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
    dppsi_bob_sub_sampling=0.9,
    dppsi_epsilon=3,
    progress_callbacks: Callable[[str, ProgressData], None] = None,
    callbacks_interval_ms: int = 5 * 1000,
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
                    dppsi_bob_sub_sampling,
                    dppsi_epsilon,
                    progress_callbacks,
                    callbacks_interval_ms,
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
    preprocess_path: Union[str, Dict[Device, str]] = None,
    ecdh_secret_key_path=None,
    dppsi_bob_sub_sampling=0.9,
    dppsi_epsilon=3,
    progress_callbacks: Callable[[str, ProgressData], None] = None,
    callbacks_interval_ms: int = 5 * 1000,
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

    if isinstance(preprocess_path, Dict):
        if isinstance(input_path, Dict):
            assert (
                preprocess_path.keys() == input_path.keys() == output_path.keys()
            ), f'mismatch key & input_path & out_path devices'
            for dev in preprocess_path.keys():
                assert (
                    dev.party in device.actors
                ), f'key {dev} not co-located with {device}'

    res = []
    if isinstance(input_path, str):
        assert isinstance(
            output_path, str
        ), f'input_path and output_path must be same types'
        for actor in device.actors.values():
            k = key[actor] if isinstance(key, Dict) else key
            p = (
                preprocess_path[actor]
                if isinstance(preprocess_path, Dict)
                else preprocess_path
            )
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
                    p,
                    ecdh_secret_key_path,
                    dppsi_bob_sub_sampling,
                    dppsi_epsilon,
                    progress_callbacks,
                    callbacks_interval_ms,
                )
            )
    else:
        for dev, ipath in input_path.items():
            opath = output_path[dev]
            actor = device.actors[dev.party]
            k = key[dev] if isinstance(key, Dict) else key
            p = (
                preprocess_path[dev]
                if isinstance(preprocess_path, Dict)
                else preprocess_path
            )
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
                    p,
                    ecdh_secret_key_path,
                    dppsi_bob_sub_sampling,
                    dppsi_epsilon,
                    progress_callbacks,
                    callbacks_interval_ms,
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
    bucket_size=1 << 20,
    curve_type="CURVE_25519",
    progress_callbacks: Callable[[str, ProgressData], None] = None,
    callbacks_interval_ms: int = 5 * 1000,
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
                    bucket_size,
                    curve_type,
                    progress_callbacks,
                    callbacks_interval_ms,
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
    bucket_size=1 << 20,
    curve_type="CURVE_25519",
    progress_callbacks: Callable[[str, ProgressData], None] = None,
    callbacks_interval_ms: int = 5 * 1000,
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
        for actor in device.actors.values():
            k = key[actor] if isinstance(key, Dict) else key
            res.append(
                actor.psi_join_csv.remote(
                    k,
                    input_path,
                    output_path,
                    receiver,
                    join_party,
                    protocol,
                    bucket_size,
                    curve_type,
                    progress_callbacks,
                    callbacks_interval_ms,
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
                    bucket_size,
                    curve_type,
                    progress_callbacks,
                    callbacks_interval_ms,
                )
            )

    # wait for all tasks done
    return sfd.get(res)


@register(DeviceType.SPU)
def pir_setup(
    device: SPU,
    server: str,
    input_path: str,
    key_columns: Union[str, List[str]],
    label_columns: Union[str, List[str]],
    oprf_key_path: str,
    setup_path: str,
    num_per_query: int,
    label_max_len: int,
    protocol="KEYWORD_PIR_LABELED_PSI",
):
    assert isinstance(device, SPU), f'device must be SPU device'
    assert isinstance(server, str), f'server must be str'
    assert isinstance(input_path, str), f'input_path must be str'
    assert isinstance(
        key_columns, (str, List, Dict)
    ), f'invalid key_columns, must be str of list of str or dict of list str'
    assert isinstance(
        label_columns, (str, List, Dict)
    ), f'invalid label_columns, must be str of list of str or dict of list str'
    assert isinstance(oprf_key_path, str), f'oprf_key_path must be str '
    assert isinstance(setup_path, str), f'setup_path must be str '
    assert isinstance(num_per_query, int), f'num_per_query must be int'
    assert isinstance(label_max_len, int), f'label_max_len must be int'

    assert server in device.actors.keys(), f'invalid server party name {server}'

    res = []

    actor = device.actors[server]
    res.append(
        actor.pir_setup.remote(
            server,
            input_path,
            key_columns,
            label_columns,
            oprf_key_path,
            setup_path,
            num_per_query,
            label_max_len,
            protocol,
        )
    )

    # wait for all tasks done
    return sfd.get(res)


@register(DeviceType.SPU)
def pir_query(
    device: SPU,
    server: str,
    config: Dict[Device, Dict],
    protocol="KEYWORD_PIR_LABELED_PSI",
):
    assert isinstance(device, SPU), f'device must be SPU device'
    assert isinstance(server, str), f'server must be str'
    assert isinstance(config, Dict), f'config must be str'

    assert server in device.actors.keys(), f'invalid server party name {server}'

    assert 2 == len(
        device.actors
    ), f'unexpected number({len(device.actors)}) of partys, should be 2'

    res = []
    for dev, iconfig in config.items():
        actor = device.actors[dev.party]
        res.append(
            actor.pir_query.remote(
                server,
                iconfig,
                protocol,
            )
        )

    # wait for all tasks done
    return sfd.get(res)


@register(DeviceType.SPU)
def pir_memory_query(
    device: SPU,
    server: str,
    config: Dict[Device, Dict],
    protocol="KEYWORD_PIR_LABELED_PSI",
):
    assert isinstance(device, SPU), f'device must be SPU device'
    assert isinstance(server, str), f'server must be str'
    assert isinstance(config, Dict), f'config must be str'

    assert server in device.actors.keys(), f'invalid server party name {server}'

    assert 2 == len(
        device.actors
    ), f'unexpected number({len(device.actors)}) of partys, should be 2'

    res = []
    for dev, iconfig in config.items():
        actor = device.actors[dev.party]
        res.append(
            actor.pir_memory_query.remote(
                server,
                iconfig,
                protocol,
            )
        )

    # wait for all tasks done
    return sfd.get(res)
