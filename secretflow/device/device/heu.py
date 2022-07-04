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

import numpy as np
import ray
from secretflow.device.device.spu import PyTreeLeaf
from secretflow.utils import ndarray_bigint
from secretflow.utils.errors import PartyNotFoundError

import spu
from heu import phe

from .base import Device, DeviceType
from .heu_object import HeCiphertext, HeuInstanceCollection
from .type_traits import (
    heu_datatype_to_numpy,
    heu_datatype_to_spu,
    spu_fxp_precision,
    spu_fxp_size,
)


class HEUActor:
    def __init__(self, heu_id, party: str, hekit, cleartext_type: np.dtype, encoder):
        """Init heu actor class
        Args:
            heu_id: Heu instance id, globally unique
            party: The party id
            hekit: phe.HeKit for sk_keeper or phe.DestinationHeKit for evaluator
            encoder: Encode cleartext (float value) to plaintext (big int value).
                available encoders:
                    - phe.PlainEncoder
        """
        self.heu_id = heu_id
        self.party = party
        self.hekit = hekit
        self.encryptor = hekit.encryptor()
        self.evaluator = hekit.evaluator()
        self.cleartext_type = cleartext_type
        self.encoder = encoder
        # register_self
        HeuInstanceCollection.meta[heu_id] = self

    def getitem(self, data, item):
        """Delegate of numpy ndarray.__getitem___()"""
        return data[item]

    def sum(self, data, *args, **kwargs):
        """Delegate of numpy ndarray.sum()"""
        assert isinstance(
            data, np.ndarray
        ), f"data must be np.ndarray type, real type={type(data)}"
        assert (
            sum(data.shape) > 0
        ), f"You cannot sum an empty ndarray, data.shape={data.shape}"

        return data.sum(*args, **kwargs)

    def _encrypt_scalar(self, scalar):
        if not isinstance(scalar, phe.Plaintext):
            scalar = self.encoder.encode(scalar)
        return HeCiphertext(self.encryptor.encrypt(scalar), self.heu_id)

    def _encrypt_scalar_with_audit(self, scalar):
        if not isinstance(scalar, phe.Plaintext):
            scalar = self.encoder.encode(scalar)
        c, a = self.encryptor.encrypt_with_audit(scalar)
        return HeCiphertext(c, self.heu_id), a

    def encrypt(self, data, heu_audit_log: str = None):
        """Encrypt data

        If the data has already been encoded, the data will be encrypted directly,
        you don't have to worry about the data being encoded repeatedly

        Even if the data has been encrypted, you still need to pass in the
        encoder param, because decryption will use it

        Args:
            data: The data to be encrypted

        Returns:
            The encrypted ndarray data
        """
        data = data if isinstance(data, np.ndarray) else np.array(data)
        if data.dtype != np.object and self.cleartext_type != "DT_FXP":
            assert np.issubdtype(data.dtype, np.integer), (
                f"This HEU (id={self.heu_id}) only supports integers, "
                f"if you want to encrypt floating numbers, please modify the "
                f"initial configuration of HEU. data_type=({data.dtype})"
            )
        if heu_audit_log:
            c, a = np.vectorize(lambda x: self._encrypt_scalar_with_audit(x))(data)
            np.save(heu_audit_log, a)
            return c
        else:
            return np.vectorize(lambda x: self._encrypt_scalar(x))(data)

    def do_math_op(self, fn, *args, **kwargs):
        """perform math operation
        Args:
            fn: numpy functions, such as np.add, np.subtract, np.multiply
        Returns:
            numpy ndarray of HeCiphertext
        """
        return fn(*args, **kwargs)


@ray.remote
class HEUSkKeeper(HEUActor):
    def __init__(self, heu_id, config, cleartext_type: np.dtype, encoder):
        assert 'he_parameters' in config, f"missing field 'he_parameters' in heu config"
        param: dict = config['he_parameters']

        assert 'key_pair' in param, f"missing field 'key_pair' in heu config"
        assert (
            'generate' in param['key_pair']
        ), f"missing field 'generate' in heu config"

        self.hekit = phe.setup(
            param.get("schema", "ou"),
            param['key_pair']['generate'].get('bit_size', 2048),
        )
        super().__init__(
            heu_id, config['sk_keeper']['party'], self.hekit, cleartext_type, encoder
        )

    def public_key(self):
        return self.hekit.public_key()

    def _decrypt_scalar(self, scalar: HeCiphertext, decode_as_int, skip_decode=False):
        assert isinstance(
            scalar, HeCiphertext
        ), f'Decrypt: scalar must be HeCiphertext type, real type is {type(scalar)} '

        pt = self.hekit.decryptor().decrypt(scalar.ct)
        if skip_decode:
            return pt

        return self.encoder.decode_int(pt) if decode_as_int else self.encoder.decode(pt)

    def decrypt(self, data: np.ndarray):
        """Decrypt data"""
        return np.vectorize(
            lambda x: self._decrypt_scalar(x, self.cleartext_type != "DT_FXP")
        )(data)

    def h2a_decrypt_make_share(
        self, var_name, data_with_mask: np.ndarray, spu_field_type
    ):
        """H2A: Decrypt the masked data array"""
        # decrypt without decode
        data_with_mask = ndarray_bigint.BigintNdArray(
            [
                int(self._decrypt_scalar(x, True, skip_decode=True))
                for x in data_with_mask.flatten()
            ],
            data_with_mask.shape,
        )

        # ValueProto: see spu.proto in SPU repo for details.
        proto = spu.ValueProto()
        proto.visibility = spu.Visibility.VIS_SECRET
        proto.data_type = heu_datatype_to_spu(self.cleartext_type)
        proto.storage_type = f"semi2k.AShr<{spu.FieldType.Name(spu_field_type)}>"
        proto.shape.dims.extend(data_with_mask.shape)
        proto.content = data_with_mask.to_bytes(spu_fxp_size(spu_field_type))
        return {var_name: proto}


@ray.remote
class HEUEvaluator(HEUActor):
    def __init__(
        self, heu_id, party: str, config, pk, cleartext_type: np.dtype, encoder
    ):
        self.config = config
        self.hekit = phe.setup(pk)
        super().__init__(heu_id, party, self.hekit, cleartext_type, encoder)

    def dump(self, data, path):
        assert isinstance(data, (HeCiphertext, np.ndarray)), (
            f'value must be HeCiphertext array, ' f'got {type(data)} instead.'
        )
        data = data if isinstance(data, np.ndarray) else np.array(data)
        np.save(path, np.vectorize(lambda x: str(x))(data))

        import cloudpickle as pickle

        pk = self.hekit.public_key()
        with open(f"{path}.pk.pickle", "wb") as f:
            pickle.dump(pk, f)

    def a2h_sum_shards(self, *shards):
        """A2H: get sum of arithmetic shares"""
        res = shards[0]
        for shard in shards[1:]:
            res += shard
        return res

    def h2a_make_share(self, var_name, data, evaluator_parties, spu_field_type):
        """H2A: make share of data, runs on the side (party) where the data resides

        Args:
            var_name: variable name in SPU, globally unique
            data: HeCiphertext array
            evaluator_parties:
            spu_field_type:

        Returns:
            Dynamical number of return values, equal to len(evaluator_parties) + 2
            Return: spu_meta_info, sk_keeper's shard, and each evaluator's shard
        """
        # This import must be placed inside the function,
        # otherwise ray cannot serialize the actor
        # https://docs.ray.io/en/releases-1.8.0/using-ray-with-tensorflow.html

        assert isinstance(data, (HeCiphertext, np.ndarray)), (
            f'value must be HeCiphertext array, ' f'got {type(data)} instead.'
        )
        data = data if isinstance(data, np.ndarray) else np.array(data)

        bound = int(self.hekit.public_key().plaintext_bound()) / 2
        masks = [
            ndarray_bigint.randint(data.shape, -bound, bound) for _ in evaluator_parties
        ]
        data_with_mask = data
        for m in masks:
            # convert data_with_mask to phe.Plaintext to avoid double encoding
            data_with_mask = data_with_mask - np.vectorize(lambda x: phe.Plaintext(x))(
                m.to_numpy()
            )

        # convert mask to ValueProto
        # ValueProto: see spu.proto in SPU repo for details.
        masks_proto = []
        for mask in masks:
            proto = spu.ValueProto()
            proto.visibility = spu.Visibility.VIS_SECRET
            proto.data_type = heu_datatype_to_spu(self.cleartext_type)
            proto.storage_type = f"semi2k.AShr<{spu.FieldType.Name(spu_field_type)}>"
            proto.shape.dims.extend(mask.shape)
            proto.content = mask.to_bytes(spu_fxp_size(spu_field_type))
            masks_proto.append({var_name: proto})

        spu_meta = PyTreeLeaf(
            var_name,
            spu.Visibility.VIS_SECRET,
            heu_datatype_to_numpy(self.cleartext_type),
            data.shape,
        )
        # Because Flake8 is very stupid, so we return a list instead of a tuple
        # If we return a tuple, Flake8 will say there is a syntax error. (・◇・)
        return [spu_meta, data_with_mask, *masks_proto]


class HEU(Device):
    """Homomorphic encryption device"""

    def __init__(self, config: dict, spu_field_type: str):
        """Initialize HEU

        Args:
            config: HEU init config, for example:
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
                            Since HeCiphertext cannot obtain HeObject objects on the Python
                            side, it is temporarily placed in the HEU instance and will be
                            fixed after the he-device logic lowering to C++.
                        'encoding': {
                            # DT_I1, DT_I8, DT_I16, DT_I32, DT_I64 or DT_FXP (default)
                            'cleartext_type': "DT_FXP"
                        }
                        'he_parameters': {
                            'schema': 'ou',
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
        if 'encoding' in config:
            self.cleartext_type = config['encoding'].get("cleartext_type", "DT_FXP")

        plain_encoder_scale = 1
        if self.cleartext_type == "DT_FXP":
            plain_encoder_scale <<= spu_fxp_precision(spu_field_type)
        self.encoder = phe.PlainEncoder(plain_encoder_scale)
        self.init()

    def init(self):
        assert (
            'sk_keeper' in self.config
        ), f"The current version does not support HEU standalone deployment mode"
        assert (
            'evaluators' in self.config and len(self.config['evaluators']) > 0
        ), f"The current version does not support HEU standalone deployment mode"

        heu_id = id(self)
        self.sk_keeper = HEUSkKeeper.options(
            resources={self.config['sk_keeper']['party']: 1}
        ).remote(heu_id, self.config, self.cleartext_type, self.encoder)

        pk = self.sk_keeper.public_key.remote()
        for cfg in self.config['evaluators']:
            self.evaluators[cfg['party']] = HEUEvaluator.options(
                resources={cfg['party']: 1}
            ).remote(
                heu_id, cfg['party'], self.config, pk, self.cleartext_type, self.encoder
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

    def __call__(self, fn, *, num_returns=None, static_argnames=None):
        raise NotImplementedError("Heu function call is not implemented")
