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

from heu import numpy as hnp
from spu import spu_pb2

from secretflow.device import (
    HEU,
    PYU,
    SPU,
    DeviceObject,
    DeviceType,
    HEUObject,
    PYUObject,
    SPUObject,
    register,
)
from secretflow.device.device.base import register_to
from secretflow.device.device.heu import HEUMoveConfig


@register_to(DeviceType.HEU, DeviceType.HEU)
def heu_to_heu(self: HEUObject, heu: HEU, config: HEUMoveConfig = None):
    assert isinstance(heu, HEU), f'Expect an HEU but got {type(heu)}.'
    if config is None:
        config = HEUMoveConfig()

    if self.device is heu:
        return heu_to_same_heu(self, config)
    else:
        return heu_to_other_heu(self, heu, config)


@register_to(DeviceType.HEU, DeviceType.PYU)
def heu_to_pyu(self: HEUObject, pyu: PYU, config: HEUMoveConfig = None):
    assert isinstance(pyu, PYU), f'Expect a PYU but got {type(pyu)}.'
    if config is None:
        config = HEUMoveConfig()

    # heu -> heu(sk_keeper)
    if self.location != pyu.party:
        config.heu_dest_party = pyu.party
        self = self.to(self.device, config)

    # below is pure local operation
    if self.is_plain:
        cleartext = self.device.get_participant(self.location).decode.remote(
            self.data, config.heu_encoder
        )
        return PYUObject(pyu, cleartext)

    assert (
        pyu.party == self.device.sk_keeper_name()
    ), f'Can not convert to PYU device {pyu.party} without secret key'

    # HEU -> PYU: Decrypt
    cleartext = self.device.sk_keeper.decrypt_and_decode.remote(
        self.data, config.heu_encoder
    )
    return PYUObject(pyu, cleartext)


@register_to(DeviceType.HEU, DeviceType.SPU)
def heu_to_spu(self: HEUObject, spu: SPU):
    assert isinstance(spu, SPU), f'Expect an SPU but got {type(spu)}.'
    heu = self.device

    assert (
        heu.sk_keeper_name() in spu.actors.keys()
    ), f'SPU not exist in {heu.sk_keeper_name()}'

    heu_parties = list(heu.evaluator_names()) + [heu.sk_keeper_name()]
    assert set(spu.actors.keys()).issubset(
        heu_parties
    ), f'Mismatch SPU and HEU parties, spu: {list(spu.actors.keys())}, heu:{heu_parties}'

    evaluator_parties = [ev for ev in heu.evaluator_names() if ev in spu.actors.keys()]

    # protocol is restricted to SEMI2K.
    assert spu.conf.protocol == spu_pb2.SEMI2K

    res = (
        heu.get_participant(self.location)
        .h2a_make_share.options(num_returns=len(evaluator_parties) + 3)
        .remote(
            self.data,
            evaluator_parties,
            spu.conf.protocol,
            spu.conf.field,
            0,
        )
    )

    meta, sk_keeper_data, io_info, chunks = (
        res[0],
        res[1],
        res[2],
        res[3:],
    )

    # sk_keeper: set data_with_mask as shard
    sk_keeper_chunk = heu.sk_keeper.h2a_decrypt_make_share.remote(
        sk_keeper_data, spu.conf.field
    )

    # make sure sk_keeper_data would be sent to the correct spu actor.
    spu_actor_idx_for_keeper = -1
    for idx, name in enumerate(spu.actors.keys()):
        if name == heu.sk_keeper_name():
            spu_actor_idx_for_keeper = idx
            break

    assert (
        spu_actor_idx_for_keeper != -1
    ), f"couldn't find {heu.sk_keeper_name()} in spu actor list."

    chunks.insert(spu_actor_idx_for_keeper, sk_keeper_chunk)

    return SPUObject(spu, meta, spu.infeed_shares(io_info, chunks))


# Data flows inside the HEU, across network
def heu_to_same_heu(self: HEUObject, config: HEUMoveConfig):
    if self.location == config.heu_dest_party:
        return self  # nothing to do

    if self.is_plain:
        # encrypt and send
        ct = self.device.get_participant(self.location).encrypt.remote(
            self.data, config.heu_audit_log
        )
        return HEUObject(self.device, ct, config.heu_dest_party, is_plain=False)
    else:
        # directly send
        return HEUObject(self.device, self.data, config.heu_dest_party, is_plain=False)


# The two HEU have different pk/sk
def heu_to_other_heu(self: DeviceObject, dest_device: HEU, config: HEUMoveConfig):
    raise NotImplementedError("Heu object cannot flow across HEUs")


def _binary_op(self: HEUObject, other: HEUObject, op) -> 'HEUObject':
    assert isinstance(other, HEUObject)
    assert self.location == other.location, (
        f"Heu objects that are not on the same node cannot perform operations, "
        f"left:{self.location}, right:{other.location}"
    )

    data = self.device.get_participant(self.location).do_binary_op.remote(
        op, self.data, other.data
    )
    return HEUObject(self.device, data, self.location, self.is_plain and other.is_plain)


@register(DeviceType.HEU)
def add(self: HEUObject, other):
    return _binary_op(self, other, hnp.Evaluator.add.__name__)


@register(DeviceType.HEU)
def sub(self: HEUObject, other):
    return _binary_op(self, other, hnp.Evaluator.sub.__name__)


@register(DeviceType.HEU)
def mul(self: HEUObject, other):
    return _binary_op(self, other, hnp.Evaluator.mul.__name__)


@register(DeviceType.HEU)
def matmul(self: HEUObject, other):
    return _binary_op(self, other, hnp.Evaluator.matmul.__name__)
