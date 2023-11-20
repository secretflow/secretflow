# Copyright 2023 Ant Group Co., Ltd.
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
import spu.libspu.link as link
from typing import Dict, Any
from secretflow.ic.proxy.serializer import serialize, deserialize


class LinkProxy:
    self_party = None
    all_parties = None

    _link = None
    _parties_rank = None

    @classmethod
    def init(
        cls,
        addresses: Dict,
        self_party: str,
    ):
        cls._parties_rank = {party: i for i, party in enumerate(addresses)}
        cls.self_party = self_party
        cls.all_parties = list(addresses.keys())

        desc = link.Desc()
        for party, addr in addresses.items():
            desc.add_party(party, addr)

        self_rank = cls.all_parties.index(self_party)

        cls._link = link.create_brpc(desc, self_rank)

    @classmethod
    def send_raw(cls, dest_party: str, msg_bytes: bytes):
        rank = cls._parties_rank[dest_party]
        cls._link.send_async(rank, msg_bytes)

    @classmethod
    def recv_raw(cls, src_party: str) -> bytes:
        rank = cls._parties_rank[src_party]
        return cls._link.recv(rank)

    @classmethod
    def send(cls, dest_party: str, data: Any):
        msg_bytes = serialize(data)
        cls.send_raw(dest_party, msg_bytes)
        logging.debug(f'send type {type(data)} to {dest_party}')

    @classmethod
    def recv(cls, src_party: str) -> Any:
        msg_bytes = cls.recv_raw(src_party)
        data = deserialize(msg_bytes)
        logging.debug(f'recv type {type(data)} from {src_party}')
        return data

    @classmethod
    def stop(cls):
        cls._link.stop_link()
