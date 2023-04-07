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

import socket
from contextlib import closing
from typing import Any, Dict, List, Tuple, cast

import spu

DEFAULT_SEMI2K_RUNTIME_CONFIG = {
    'protocol': spu.spu_pb2.SEMI2K,
    'field': spu.spu_pb2.FM128,
}

DEFAULT_ABY3_RUNTIME_CONFIG = {
    'protocol': spu.spu_pb2.ABY3,
    'field': spu.spu_pb2.FM128,
}


def unused_tcp_port() -> int:
    """Return an unused port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return cast(int, sock.getsockname()[1])


def cluster_def(parties: List[str], runtime_config=None) -> Dict[str, Any]:
    """Generate SPU cluster_def for testing purposes.

    Args:
        parties (List[str]): parties of SPU devices. Must be more than 1 party,
        runtime_config (optional): runtime_config of SPU device.
            More details refer to
                `SPU runtime config <https://www.secretflow.org.cn/docs/spu/en/reference/runtime_config.html>`_.
            Defaults to None and use default runtime config.
                1. If 3 parties are present, protocol would be set to ABY3 and field to FM128.
                2. Otherwise, protocol would be set to SEMI2k and field to FM128.
                Other options are using default values.

    Returns:
        Dict[str, Any]: An SPU cluster_def to initiate an SPU device.
    """
    assert (
        isinstance(parties, (Tuple, List)) and len(parties) >= 2
    ), 'number of parties should be >= 2'
    assert len(set(parties)) == len(parties), f'duplicated parties {parties}'

    if not runtime_config:
        if len(parties) == 3:
            runtime_config = DEFAULT_ABY3_RUNTIME_CONFIG
        else:
            runtime_config = DEFAULT_SEMI2K_RUNTIME_CONFIG

    if runtime_config['protocol'] == spu.spu_pb2.ABY3:
        assert len(parties) == 3, 'ABY3 only supports 3PC.'

    cdef = {
        'nodes': [],
        'runtime_config': runtime_config,
    }

    for i, party in enumerate(parties):
        cdef['nodes'].append(
            {
                'party': party,
                'address': f'127.0.0.1:{unused_tcp_port()}',
            }
        )

    return cdef


def heu_config(sk_keeper: str, evaluators: List[str]):
    return {
        'sk_keeper': {'party': sk_keeper},
        'evaluators': [{'party': evaluator} for evaluator in evaluators],
        'mode': 'PHEU',
        'he_parameters': {
            'schema': 'paillier',
            'key_pair': {'generate': {'bit_size': 2048}},
        },
        'encoding': {
            'cleartext_type': 'DT_I32',
            'encoder': "IntegerEncoder",
            'encoder_args': {"scale": 1},
        },
    }
