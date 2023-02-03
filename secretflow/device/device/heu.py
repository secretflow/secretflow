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
from functools import reduce
from pathlib import Path
from typing import Union
import jax.tree_util
import ray
import cloudpickle as pickle
import numpy as np
import spu
from heu import numpy as hnp
from heu import phe

import secretflow.distributed as sfd
from secretflow.utils.errors import PartyNotFoundError

from .base import Device, DeviceType
from .spu import SPUValueMeta
from .type_traits import (
    heu_datatype_to_numpy,
    heu_datatype_to_spu,
    spu_fxp_precision,
    spu_fxp_size,
)


class HEUActor:
    def __init__(
        self,
        heu_id,
        party: str,
        hekit: Union[hnp.HeKit, hnp.DestinationHeKit],
        cleartext_type: np.dtype,
        encoder,
    ):
        """Init heu actor class

        Args:
            heu_id: Heu instance id, globally unique
            party: The party id
            hekit: hnp.HeKit for sk_keeper or hnp.DestinationHeKit for evaluator
            encoder: Encode cleartext (float value) to plaintext (big int value).
                available encoders:
                    - phe.IntegerEncoder
                    - phe.FloatEncoder
                    - phe.BigintEncoder
                    - phe.BatchEncoder
        """
        self.heu_id = heu_id
        self.party = party
        self.hekit = hekit
        self.encryptor = hekit.encryptor()
        self.evaluator = hekit.evaluator()
        self.cleartext_type = cleartext_type
        self.encoder = encoder

    def getitem(self, data, item):
        """Delegate of hnp ndarray.__getitem___()"""
        item = jax.tree_util.tree_map(
            lambda x: ray.get(x) if isinstance(x, ray.ObjectRef) else x,
            item,
        )
        item = jax.tree_util.tree_map(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x, item
        )

        return data[item]

    def setitem(self, data, key, value):
        """Delegate of hnp ndarray.__setitem___()"""
        if isinstance(key, np.ndarray):
            key = key.tolist()

        data[key] = value

    def sum(self, data):
        """sum of data elements"""
        assert isinstance(
            data, (hnp.PlaintextArray, hnp.CiphertextArray)
        ), f"data must be hnp.ndarray type, real type={type(data)}"
        assert (
            data.size > 0
        ), f"You cannot sum an empty ndarray, data.shape={data.rows}x{data.cols}"

        return self.evaluator.sum(data)

    def encode(self, data: np.ndarray, edr=None):
        """encode cleartext to plaintext

        Args:
            data: cleartext
            edr: encoder
        """
        if isinstance(data, (hnp.PlaintextArray, hnp.CiphertextArray)):
            return

        return self.hekit.array(data, self.encoder if edr is None else edr)

    def decode(self, data: hnp.PlaintextArray, edr=None):
        """decode plaintext to cleartext

        Args:
            data: plaintext
            edr: encoder
        """
        if edr is None:
            edr = self.encoder
        if isinstance(
            edr,
            (
                phe.BigintEncoderParams,
                phe.IntegerEncoderParams,
                phe.FloatEncoderParams,
                phe.BigintEncoderParams,
            ),
        ):
            edr = edr.instance(self.hekit.get_schema())
        if isinstance(data, hnp.PlaintextArray):
            return data.to_numpy(edr)

        if isinstance(data, phe.Plaintext):
            return edr.decode(data)

        raise AssertionError(f"heu can not decode {type(data)} type")

    def encrypt(
        self, data: hnp.PlaintextArray, heu_audit_log: str = None
    ) -> hnp.CiphertextArray:
        """Encrypt data

        If the data has already been encoded, the data will be encrypted directly,
        you don't have to worry about the data being encoded repeatedly

        Even if the data has been encrypted, you still need to pass in the
        encoder param, because decryption will use it

        Args:
            data: The data to be encrypted
            heu_audit_log: file path to log audit info

        Returns:
            The encrypted ndarray data
        """
        assert isinstance(
            data, hnp.PlaintextArray
        ), f"data must be hnp.ndarray type, real type={type(data)}"
        if heu_audit_log:
            cm, audit = self.encryptor.encrypt_with_audit(data)
            with open(heu_audit_log, "wb") as f:
                pickle.dump(audit, f)
            return cm

        return self.encryptor.encrypt(data)

    def do_binary_op(self, fn_name, data1, data2):
        """perform math operation
        Args:
            fn: hnp.Evaluator functions, such as hnp.Evaluator.add, hnp.Evaluator.sub
        Returns:
            numpy ndarray of HeCiphertext
        """
        fn = getattr(hnp.Evaluator, fn_name)
        return fn(self.evaluator, data1, data2)


class HEUSkKeeper(HEUActor):
    def __init__(self, heu_id, config, cleartext_type: np.dtype, encoder):
        assert 'he_parameters' in config, f"missing field 'he_parameters' in heu config"
        param: dict = config['he_parameters']

        assert 'key_pair' in param, f"missing field 'key_pair' in heu config"
        assert (
            'generate' in param['key_pair']
        ), f"missing field 'generate' in heu config"

        self.hekit = hnp.setup(
            param.get("schema", "paillier"),
            param['key_pair']['generate'].get('bit_size', 2048),
        )
        super().__init__(
            heu_id, config['sk_keeper']['party'], self.hekit, cleartext_type, encoder
        )

    def public_key(self):
        return self.hekit.public_key()

    def dump_pk(self, path):
        """Dump public key to the specified file."""
        pk = self.hekit.public_key()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(pk, f)

    def decrypt(self, data) -> Union[phe.Plaintext, hnp.PlaintextArray]:
        """Decrypt data: ciphertext -> plaintext"""
        if isinstance(data, hnp.CiphertextArray):
            return self.hekit.decryptor().decrypt(data)

        if isinstance(data, phe.Ciphertext):
            return self.hekit.decryptor().phe.decrypt(data)

        raise AssertionError(f"heu can not decrypt {type(data)} type")

    def decrypt_and_decode(self, data: hnp.CiphertextArray, edr=None):
        """Decrypt data: ciphertext -> cleartext

        Args:
            data: ciphertext
            edr: encoder
        """
        return self.decode(self.decrypt(data), edr)

    def h2a_decrypt_make_share(
        self, data_with_mask: hnp.CiphertextArray, spu_field_type
    ):
        """H2A: Decrypt the masked data array"""
        # decrypt without decode
        data_with_mask = self.decrypt(data_with_mask)
        byte_content = data_with_mask.to_bytes(spu_fxp_size(spu_field_type), 'little')
        # ValueProto: see spu.proto in SPU repo for details.
        proto = spu.ValueProto()
        proto.visibility = spu.Visibility.VIS_SECRET
        proto.data_type = heu_datatype_to_spu(self.cleartext_type)
        proto.storage_type = f"semi2k.AShr<{spu.FieldType.Name(spu_field_type)}>"
        proto.shape.dims.extend(data_with_mask.shape)
        proto.content = byte_content
        return proto.SerializeToString()


class HEUEvaluator(HEUActor):
    def __init__(
        self, heu_id, party: str, config, pk, cleartext_type: np.dtype, encoder
    ):
        self.config = config
        self.hekit = hnp.setup(pk)
        super().__init__(heu_id, party, self.hekit, cleartext_type, encoder)

    def dump(self, data, path):
        """Dump data to file."""
        assert isinstance(data, (hnp.CiphertextArray, hnp.PlaintextArray)), (
            f'value must be hnp array, ' f'got {type(data)} instead.'
        )

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def dump_pk(self, path):
        """Dump public key to the specified file."""
        pk = self.hekit.public_key()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(pk, f)

    def a2h_sum_shards(self, *shards):
        """A2H: get sum of arithmetic shares"""
        return reduce(self.evaluator.add, shards)

    def h2a_make_share(
        self,
        data: hnp.CiphertextArray,
        evaluator_parties,
        spu_protocol,
        spu_field_type,
        spu_fxp_fraction_bits,
    ):
        """H2A: make share of data, runs on the side (party) where the data resides

        Args:
            data: HeCiphertext array
            evaluator_parties:
            spu_protocol: part of spu runtime config.
            spu_field_type: part of spu runtime config.
            spu_fxp_fraction_bits: part of spu runtime config.

        Returns:
            Dynamical number of return values, equal to len(evaluator_parties) + 2
            Return: spu_meta_info, sk_keeper's shard, and each evaluator's shard
        """
        # This import must be placed inside the function,
        # otherwise ray cannot serialize the actor
        # https://docs.ray.io/en/releases-1.8.0/using-ray-with-tensorflow.html

        assert isinstance(data, hnp.CiphertextArray), (
            f'value must be HeCiphertext array, ' f'got {type(data)} instead.'
        )

        # we should make (random + n) <= plaintext_bound,
        # so we restrict random bound to half of plaintext_bound
        bound = self.hekit.public_key().plaintext_bound() / phe.Plaintext(
            self.hekit.get_schema(), 2
        )
        masks = [hnp.random.randint(-bound, bound, data.shape)]
        data_with_mask: hnp.CiphertextArray = data
        for m in masks:
            data_with_mask = self.evaluator.sub(data_with_mask, m)

        # convert mask to ValueProto
        # ValueProto: see spu.proto in SPU repo for details.
        masks_value = []
        for mask in masks:
            proto = spu.ValueProto()
            proto.visibility = spu.Visibility.VIS_SECRET
            proto.data_type = heu_datatype_to_spu(self.cleartext_type)
            proto.storage_type = f"semi2k.AShr<{spu.FieldType.Name(spu_field_type)}>"
            proto.shape.dims.extend(tuple(mask.shape))
            proto.content = mask.to_bytes(spu_fxp_size(spu_field_type), 'little')
            masks_value.append(proto.SerializeToString())

        value_meta = SPUValueMeta(
            data.shape,
            heu_datatype_to_numpy(self.cleartext_type),
            spu.Visibility.VIS_SECRET,
            spu_protocol,
            spu_field_type,
            spu_fxp_fraction_bits,
        )

        # Because Flake8 is very stupid, so we return a list instead of a tuple
        # If we return a tuple, Flake8 will say there is a syntax error. (・◇・)
        return [
            value_meta,
            data_with_mask,
            *masks_value,
        ]


class HEU(Device):
    """Homomorphic encryption device"""

    def __init__(self, config: dict, spu_field_type):
        """Initialize HEU

        Args:
            config: HEU init config, for example

                .. code:: python

                    {
                        'sk_keeper': {
                            'party': 'alice'
                        },
                        'evaluators': [{
                            'party': 'bob'
                        }],
                        # The HEU working mode, choose from PHEU / LHEU / FHEU_ROUGH / FHEU
                        'mode': 'PHEU',
                        # TODO: cleartext_type should be migrated to HeObject.
                        'encoding': {
                            # DT_I1, DT_I8, DT_I16, DT_I32, DT_I64 or DT_FXP (default)
                            'cleartext_type': "DT_FXP"
                            # see https://www.secretflow.org.cn/docs/heu/en/getting_started/quick_start.html#id3 for detail
                            # available encoders:
                            #     - IntegerEncoder: Plaintext = Cleartext * scale
                            #     - FloatEncoder (default): Plaintext = Cleartext * scale
                            #     - BigintEncoder: Plaintext = Cleartext
                            #     - BatchEncoder: Plaintext = Pack[Cleartext, Cleartext]
                            'encoder': 'FloatEncoder'
                        }
                        'he_parameters': {
                            'schema': 'paillier',
                            'key_pair': {
                                'generate': {
                                    'bit_size': 2048,
                                },
                            }
                        }
                    }
            spu_field_type: Field type in spu,
                Device.to operation requires the data scale of HEU to be aligned with
                SPU
        """
        super().__init__(DeviceType.HEU)

        config.setdefault('mode', 'PHEU')
        assert (
            config['mode'] == 'PHEU'
        ), f'HEU working mode {config["mode"]} not supported now'

        self.sk_keeper = None
        self.evaluators = {}
        self.config = config

        self.cleartext_type = "DT_FXP"
        default_scale = 1 << spu_fxp_precision(spu_field_type)
        assert 'he_parameters' in config, f"missing field 'he_parameters' in heu config"
        param: dict = config['he_parameters']
        schema = phe.parse_schema_type(param.get("schema", "paillier"))
        self.schema = schema
        self.encoder = phe.FloatEncoder(schema, default_scale)
        self.scale = default_scale
        if 'encoding' in config:
            cfg = config['encoding']
            self.cleartext_type = cfg.get("cleartext_type", "DT_FXP")
            edr_args = cfg.get("encoder_args", {})
            edr_name = cfg.get("encoder", "FloatEncoder")

            if edr_name == "IntegerEncoder":
                edr_args["scale"] = edr_args.get("scale", default_scale)
                self.encoder = phe.IntegerEncoder(schema, **edr_args)
                self.scale = edr_args["scale"]
            elif edr_name == "FloatEncoder":
                edr_args["scale"] = edr_args.get("scale", default_scale)
                self.encoder = phe.FloatEncoder(schema, **edr_args)
                self.scale = edr_args["scale"]
            elif edr_name == "BigintEncoder":
                self.encoder = phe.BigintEncoder(schema)
                self.scale = 1
            elif edr_name == "BatchEncoder":
                self.encoder = phe.BatchEncoder(schema, **edr_args)
                self.scale = edr_args.get("scale", 1)
            else:
                raise AssertionError(f"Unsupported encoder type {edr_name}")

        self.init()

    def init(self):
        assert (
            'sk_keeper' in self.config
        ), f"The current version does not support HEU standalone deployment mode"
        assert (
            'evaluators' in self.config and len(self.config['evaluators']) > 0
        ), f"The current version does not support HEU standalone deployment mode"

        heu_id = id(self)
        self.sk_keeper = (
            sfd.remote(HEUSkKeeper)
            .party(self.config['sk_keeper']['party'])
            .remote(heu_id, self.config, self.cleartext_type, self.encoder)
        )

        pk = sfd.get(self.sk_keeper.public_key.remote())
        for cfg in self.config['evaluators']:
            self.evaluators[cfg['party']] = (
                sfd.remote(HEUEvaluator)
                .party(cfg['party'])
                .remote(
                    heu_id,
                    cfg['party'],
                    self.config,
                    pk,
                    self.cleartext_type,
                    self.encoder,
                )
            )

    def sk_keeper_name(self):
        return self.config['sk_keeper']['party']

    def evaluator_names(self):
        return self.evaluators.keys()

    def get_participant(self, party: str):
        """Get ray actor by name"""
        if party in self.evaluators:
            return self.evaluators[party]
        elif party == self.sk_keeper_name():
            return self.sk_keeper
        else:
            raise PartyNotFoundError(f"party {party} is not a participant in HEU")

    def has_party(self, party: str):
        return party == self.sk_keeper_name() or party in self.evaluators

    def __call__(self, fn, *, num_returns=None, static_argnames=None):
        raise NotImplementedError("Heu function call is not implemented")
