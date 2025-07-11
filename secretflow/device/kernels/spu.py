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
from secretflow.device.device.base import register_to
from secretflow.device.device.heu import HEUMoveConfig
from secretflow.utils.progress import ProgressData


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


# WARNING: you may need to wait spu to spu for following applications
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
        (
            heu.get_participant(p).encrypt.remote(shard, config.heu_audit_log)
            if p != config.heu_dest_party
            else shard
        )
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
    protocol='PROTOCOL_RR22',
    precheck_input=True,
    sort=True,
    broadcast_result=True,
    bucket_size=1 << 20,
    curve_type="CURVE_25519",
    dppsi_bob_sub_sampling=0.9,
    dppsi_epsilon=3,
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
                    dppsi_bob_sub_sampling,
                    dppsi_epsilon,
                ),
            )
        )

    return res


@register(DeviceType.SPU)
def ub_psi(
    device: SPU,
    mode: str,
    role: Dict[str, str],
    input_path: Dict[str, str],
    output_path: Dict[str, str],
    keys: Dict[str, List[str]],
    server_secret_key_path: str,
    cache_path: Dict[str, str],
    join_type: str,
    left_side: str,
    server_get_result: bool,
    client_get_result: bool,
    disable_alignment: bool,
    null_rep: str,
):
    assert isinstance(device, SPU), 'device must be SPU device'
    assert device.world_size == 2, 'only 2pc is allowed.'
    res = []
    for party, actor in device.actors.items():
        res.append(
            actor.ub_psi.remote(
                mode=mode,
                role=role[party],
                input_path=(
                    input_path[party] if input_path and party in input_path else ''
                ),
                keys=keys[party] if keys and party in keys else [],
                server_secret_key_path=server_secret_key_path,
                cache_path=(
                    cache_path[party] if cache_path and party in cache_path else ''
                ),
                server_get_result=server_get_result,
                client_get_result=client_get_result,
                disable_alignment=disable_alignment,
                output_path=(
                    output_path[party]
                    if (output_path and (party in output_path))
                    else ''
                ),
                join_type=join_type,
                left_side=left_side,
                null_rep=null_rep,
            )
        )
    # wait for all tasks done
    return sfd.get(res)


@register(DeviceType.SPU)
def psi(
    device: SPU,
    keys: Dict[str, List[str]],
    input_path: Dict[str, str],
    output_path: Dict[str, str],
    receiver: str,
    table_keys_duplicated: Dict[str, str],
    output_csv_na_rep: str,
    broadcast_result: bool = True,
    protocol: str = 'PROTOCOL_RR22',
    ecdh_curve: str = 'CURVE_FOURQ',
    advanced_join_type: str = "JOIN_TYPE_UNSPECIFIED",
    left_side: str = "ROLE_RECEIVER",
    disable_alignment: bool = False,
    bucket_size=1 << 20,
    dppsi_bob_sub_sampling=0.9,
    dppsi_epsilon=3,
):
    assert isinstance(device, SPU), 'device must be SPU device'

    assert receiver in device.actors, f'receiver {receiver} is not found in spu nodes.'

    res = []
    for party, actor in device.actors.items():
        res.append(
            actor.psi.remote(
                keys=keys[party],
                input_path=input_path[party],
                output_path=output_path[party] if party in output_path else "",
                receiver=receiver,
                table_keys_duplicated=(
                    table_keys_duplicated[party] if table_keys_duplicated else False
                ),
                output_csv_na_rep=output_csv_na_rep,
                broadcast_result=broadcast_result,
                protocol=protocol,
                ecdh_curve=ecdh_curve,
                join_type=advanced_join_type,
                left_side=left_side,
                disable_alignment=disable_alignment,
                bucket_size=bucket_size,
                dppsi_bob_sub_sampling=dppsi_bob_sub_sampling,
                dppsi_epsilon=dppsi_epsilon,
            )
        )
    # wait for all tasks done
    return sfd.get(res)


def party_shards_to_heu_plain_text(spu_obj: SPUObject, heu: HEU, party: str):
    assert party in spu_obj.device.actors.keys()
    assert heu.has_party(party)

    io_info, shares_chunk = spu_obj.device.outfeed_shares(spu_obj.shares_name)
    assert (
        len(shares_chunk) % len(spu_obj.device.actors) == 0
    ), f"{len(shares_chunk)} % {len(spu_obj.device.actors)}"
    chunks_count_per_party = int(len(shares_chunk) / len(spu_obj.device.actors))
    party_chunks = None
    for i, p in enumerate(spu_obj.device.actors.keys()):
        if p == party:
            party_chunks = shares_chunk[
                i * chunks_count_per_party : (i + 1) * chunks_count_per_party
            ]
            break
    assert party_chunks

    shards_data = spu_obj.device.actors[party].a2h.remote(
        io_info, heu.cleartext_type, heu.schema, *party_chunks
    )

    return HEUObject(heu, shards_data, party)
