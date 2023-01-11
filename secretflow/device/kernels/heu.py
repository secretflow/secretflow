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

from secretflow.device import (HEU, PYU, SPU, Device, DeviceObject, DeviceType,
                               HEUObject, PYUObject, SPUObject, register)
from secretflow.device.device.base import MoveConfig


@register(DeviceType.HEU)
def to(self: HEUObject, device: Device, config):
    assert isinstance(device, Device)

    if isinstance(device, HEU):
        if self.device is device:
            return heu_to_same_heu(self, config)
        else:
            return heu_to_other_heu(self, device, config)
    if isinstance(device, PYU):  # pure local operation
        return heu_to_pyu(self, device, config)
    if isinstance(device, SPU):
        return heu_to_spu(self, device)

    raise ValueError(f'Unexpected device type: {type(device)}')


def heu_to_spu(self: HEUObject, spu: SPU):
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
        .h2a_make_share.options(num_returns=len(evaluator_parties) + 2)
        .remote(
            self.data,
            evaluator_parties,
            spu.conf.protocol,
            spu.conf.field,
            0,
        )
    )

    meta, sk_keeper_data, refs = (
        res[0],
        res[1],
        res[2:],
    )

    # sk_keeper: set data_with_mask as shard
    sk_keeper_data = heu.sk_keeper.h2a_decrypt_make_share.remote(
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

    refs.insert(spu_actor_idx_for_keeper, sk_keeper_data)

    return SPUObject(spu, meta, spu.infeed_shares(refs))


# Data flows inside the HEU, across network
def heu_to_same_heu(self: HEUObject, config: MoveConfig):
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
def heu_to_other_heu(self: DeviceObject, dest_device: HEU, config):
    raise NotImplementedError("Heu object cannot flow across HEUs")


def heu_to_pyu(self: HEUObject, device: PYU, config: MoveConfig):
    # heu -> heu(sk_keeper)
    if self.location != device.party:
        config.heu_dest_party = device.party
        self = self.to(self.device, config)

    # below is pure local operation
    if self.is_plain:
        cleartext = self.device.get_participant(self.location).decode.remote(
            self.data, config.heu_encoder
        )
        return PYUObject(device, cleartext)

    assert (
        device.party == self.device.sk_keeper_name()
    ), f'Can not convert to PYU device {device.party} without secret key'

    # HEU -> PYU: Decrypt
    cleartext = self.device.sk_keeper.decrypt_and_decode.remote(
        self.data, config.heu_encoder
    )
    return PYUObject(device, cleartext)


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
