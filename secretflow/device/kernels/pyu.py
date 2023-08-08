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


import logging
import secrets
from typing import Any, Callable, List, Union

from spu import Visibility

import secretflow.distributed as sfd
from secretflow.device import (
    HEU,
    PYU,
    SPU,
    SPUIO,
    TEEU,
    DeviceType,
    HEUObject,
    PYUObject,
    SPUObject,
    TEEUData,
    TEEUObject,
    global_state,
    wait,
)
from secretflow.device.device.base import register_to
from secretflow.device.device.heu import HEUMoveConfig


@register_to(DeviceType.PYU, DeviceType.PYU)
def pyu_to_pyu(self: PYUObject, pyu: PYU) -> PYUObject:
    assert isinstance(pyu, PYU), f'Expect a PYU but got {type(pyu)}.'
    return PYUObject(pyu, self.data)


@register_to(DeviceType.PYU, DeviceType.SPU)
def pyu_to_spu(self: PYUObject, spu: SPU, spu_vis: str = 'secret') -> SPUObject:
    """Transfer pyuobject to the spu.

    Args:
        self: the pyuobject to transfer.
        spu: to this SPU device.
        spu_vis: optional; SPU object visibility. Value can be:
            - secret: Secret sharing with protocol spdz-2k, aby3, etc.
            - public: Public sharing, which means data will be replicated to each node.

    Returns:
        the transferred SPUObject.
    """
    assert isinstance(spu, SPU), f'Expect an SPU but got {type(spu)}'
    assert spu_vis in ('secret', 'public'), f'vis must be public or secret'

    vtype = Visibility.VIS_PUBLIC if spu_vis == 'public' else Visibility.VIS_SECRET

    def get_shares_chunk_count(data, runtime_config, world_size, vtype) -> int:
        io = SPUIO(runtime_config, world_size)
        return io.get_shares_chunk_count(data, vtype)

    def run_spu_io(data, runtime_config, world_size, vtype):
        io = SPUIO(runtime_config, world_size)
        ret = io.make_shares(data, vtype)
        return ret

    shares_chunk_count = self.device(get_shares_chunk_count)(
        self.data, spu.conf, spu.world_size, vtype
    )
    shares_chunk_count = sfd.get(shares_chunk_count.data)

    meta, io_info, *shares_chunk = self.device(
        run_spu_io, num_returns=(2 + shares_chunk_count * spu.world_size)
    )(self.data, spu.conf, spu.world_size, vtype)

    return SPUObject(
        spu,
        meta.data,
        spu.infeed_shares(io_info.data, [s.data for s in shares_chunk]),
    )


@register_to(DeviceType.PYU, DeviceType.HEU)
def pyu_to_heu(self: PYUObject, heu: HEU, config: HEUMoveConfig = None):
    assert isinstance(heu, HEU), f'Expect an HEU but got {type(heu)}'
    if config is None:
        config = HEUMoveConfig()

    if config.heu_dest_party == 'auto':
        config.heu_dest_party = list(heu.evaluator_names())[0]

    data = heu.get_participant(self.device.party).encode.remote(
        self.data, config.heu_encoder
    )
    return HEUObject(heu, data, self.device.party, True).to(heu, config)


@register_to(DeviceType.PYU, DeviceType.TEEU)
def pyu_to_teeu(
    self: PYUObject,
    teeu: TEEU,
    allow_funcs: Union[Callable, List[Callable]],
):
    """Transfer a PYUObject to TEEU.

    Transfer a PYUObject to TEEU, the main steps are:
    1) Create an authority with the specific function and enclave through
        the authority manager. A data key will be generated for next step.
    2) Use the data key to encrypt the data with AES-GCM.

    Args:
        self: the PYUObject instance.
        teeu: the TEEU.
        allow_funcs: the function(s) to allow using this PYUObject.
            Function not in this list can not use this PYUObject.

    Returns:
        A TEEUObject whose underlying data is ciphertext.
    """
    assert isinstance(teeu, TEEU), f'Expect a TEEU but got {type(teeu)}'
    logging.debug(
        f'Transfer PYU object from {self.device.party} to TEEU of {teeu.party}.'
    )

    def create_auth(
        data: Any,
        public_key: str,
        private_key: str,
        tls_cert: str,
        tls_key: str,
        allow_funcs: Union[Callable, List[Callable]],
        allow_enclaves: List[str],
        auth_host: str,
        auth_mr_enclave: str,
        auth_ca_cert: str,
        sim: bool,
    ):
        from sdc.auth_frame import AuthFrame, CredentialsConf

        if not isinstance(allow_funcs, (list, tuple)):
            allow_funcs = [allow_funcs]

        from secretflow.utils.cloudpickle import (
            code_position_independent_dumps as dumps,
        )

        allow_funcs_bytes = [dumps(func, protocol=4) for func in allow_funcs]
        if auth_ca_cert:
            credentials = CredentialsConf(
                root_ca=auth_ca_cert.encode('utf-8'),
                private_key=tls_key.encode('utf-8') if tls_key else None,
                cert_chain=tls_cert.encode('utf-8') if tls_cert else None,
            )
        else:
            credentials = None
        auth_frame = AuthFrame(
            authm_host=auth_host,
            authm_mr_enclave=auth_mr_enclave,
            conf=credentials,
            sim=sim,
        )
        data_uuid, data_key = auth_frame.create_auth(
            data=data,
            public_key_pem=public_key,
            private_key_pem=private_key,
            allow_funcs=allow_funcs_bytes,
            allow_enclaves=allow_enclaves,
        )

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        aesgcm = AESGCM(data_key)
        nonce = secrets.token_bytes(12)
        aad = data_uuid.encode('utf-8')

        import ray.cloudpickle as pickle

        encrypted_data = aesgcm.encrypt(
            nonce=nonce, data=pickle.dumps(data, protocol=4), associated_data=aad
        )

        return TEEUData(
            data=encrypted_data,
            data_uuid=data_uuid,
            nonce=nonce,
            aad=aad,
        )

    party = self.device.party
    if party == global_state.self_party():
        assert party in global_state.party_key_pairs(), (
            f'Can not find key pair of {party}, '
            'you can provide it through `party_key_pair` when calling `sf.init`'
        )
        party_key_pair = global_state.party_key_pairs()[party]
    else:
        party_key_pair = global_state.PartyKeyPair()
    party_cert = global_state.party_certs().get(party, global_state.PartyCert())
    teeu_data = self.device(create_auth)(
        data=self.data,
        public_key=party_key_pair.public_key,
        private_key=party_key_pair.private_key,
        tls_cert=party_cert.cert,
        tls_key=party_cert.key,
        allow_funcs=allow_funcs,
        allow_enclaves=[teeu.mr_enclave],
        auth_host=global_state.auth_manager_host(),
        auth_mr_enclave=global_state.auth_manager_mr_enclave(),
        auth_ca_cert=global_state.auth_manager_ca_cert(),
        sim=global_state.tee_simulation(),
    )
    wait(teeu_data)

    return TEEUObject(teeu, teeu_data.data)
