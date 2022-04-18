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
from typing import List, Tuple
from typing import cast

import ppu


def unused_tcp_port() -> int:
    """Return an unused port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return cast(int, sock.getsockname()[1])


def cluster_def(parties: List[str], protocol=ppu.ppu_pb2.SEMI2K):
    """Generate PPU cluster_def for testing"""
    assert isinstance(
        parties,
        (Tuple,
         List)) and len(parties) >= 2, f'number of parties should be >= 2'
    assert len(set(parties)) == len(parties), f'duplicated parties {parties}'
    if protocol == ppu.ppu_pb2.ABY3:
        assert len(parties) == 3, 'ABY3 only supports 3PC.'

    cdef = {
        'nodes': [],
        'runtime_config': {
            'protocol': protocol,
            'field': ppu.ppu_pb2.FM128,
            'sigmoid_mode': ppu.ppu_pb2.MM1,
        }
    }

    for i, party in enumerate(parties):
        cdef['nodes'].append({
            'party': party,
            'id': f'local:{i}',
            'address': f'127.0.0.1:{unused_tcp_port()}'
        })

    return cdef
