# Copyright 2024 The RayFed Team
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
from dataclasses import dataclass
from typing import Dict, Optional

import spu.libspu.link as link

from secretflow.utils import secure_pickle
from secretflow.utils.errors import YACLError

from ..base import CrossSiloMessageConfig, SenderReceiverProxy

logger = logging.getLogger(__name__)


@dataclass
class BrpcLinkCrossSiloMessageConfig(CrossSiloMessageConfig):
    connect_retry_times: Optional[int] = None
    connect_retry_interval_ms: Optional[int] = None
    recv_timeout_ms: Optional[int] = None
    http_timeout_ms: Optional[int] = None
    http_max_payload_size: Optional[int] = None
    throttle_window_size: Optional[int] = None
    brpc_channel_protocol: Optional[str] = None
    brpc_channel_connection_type: Optional[str] = None
    brpc_retry_count: Optional[int] = None
    brpc_retry_interval_ms: Optional[int] = None
    brpc_aggressive_retry: Optional[bool] = None

    def dump_to_link_desc(self, link_desc: link.Desc):
        if self.timeout_in_ms is not None:
            link_desc.http_timeout_ms = self.timeout_in_ms

        if self.connect_retry_times is not None:
            link_desc.connect_retry_times = self.connect_retry_times
        if self.connect_retry_interval_ms is not None:
            link_desc.connect_retry_interval_ms = self.connect_retry_interval_ms
        if self.recv_timeout_ms is not None:
            link_desc.recv_timeout_ms = self.recv_timeout_ms
        if self.http_timeout_ms is not None:
            logging.warning(
                'http_timeout_ms and timeout_ms are set at the same time, '
                f'http_timeout_ms {self.http_timeout_ms} will be used.'
            )
            link_desc.http_timeout_ms = self.http_timeout_ms
        if self.http_max_payload_size is not None:
            link_desc.http_max_payload_size = self.http_max_payload_size
        if self.throttle_window_size is not None:
            link_desc.throttle_window_size = self.throttle_window_size
        if self.brpc_channel_protocol is not None:
            link_desc.brpc_channel_protocol = self.brpc_channel_protocol
        if self.brpc_channel_connection_type is not None:
            link_desc.brpc_channel_connection_type = self.brpc_channel_connection_type
        if self.brpc_aggressive_retry is not None:
            link_desc.brpc_aggressive_retry = self.brpc_aggressive_retry
        if self.brpc_retry_count is not None:
            link_desc.brpc_retry_count = self.brpc_retry_count
        if self.brpc_retry_interval_ms is not None:
            link_desc.brpc_retry_interval_ms = self.brpc_retry_interval_ms

        if not hasattr(link_desc, 'recv_timeout_ms'):
            # set default timeout 3600s
            link_desc.recv_timeout_ms = 3600 * 1000
        if not hasattr(link_desc, 'http_timeout_ms'):
            # set default timeout 120s
            link_desc.http_timeout_ms = 120 * 1000


def _fill_link_ssl_opts(tls_config: Dict, link_ssl_opts: link.SSLOptions):
    ca_cert = tls_config['ca_cert']
    cert = tls_config['cert']
    key = tls_config['key']
    link_ssl_opts.cert.certificate_path = cert
    link_ssl_opts.cert.private_key_path = key
    link_ssl_opts.verify.ca_file_path = ca_cert
    link_ssl_opts.verify.verify_depth = 1


class BrpcLinkProxy(SenderReceiverProxy):
    def __init__(
        self,
        addresses: Dict,
        self_party: str,
        job_name: str,
        tls_config: Dict = None,
        proxy_config: Dict = None,
    ) -> None:
        logging.info(f'brpc options: {proxy_config}')
        proxy_config = BrpcLinkCrossSiloMessageConfig.from_dict(proxy_config)
        super().__init__(job_name, addresses, self_party, tls_config, proxy_config)
        self._parties_rank = {
            party: i for i, party in enumerate(self._addresses.keys())
        }
        self._rank = list(self._addresses).index(self_party)

        desc = link.Desc()
        for party, addr in self._addresses.items():
            desc.add_party(party, addr)
        if tls_config:
            _fill_link_ssl_opts(tls_config, desc.server_ssl_opts)
            _fill_link_ssl_opts(tls_config, desc.client_ssl_opts)
        if isinstance(proxy_config, BrpcLinkCrossSiloMessageConfig):
            proxy_config.dump_to_link_desc(desc)
        self._desc = desc

        self._all_data = {}

    def concurrent(self):
        return False

    def start(self):
        try:
            self._linker = link.create_brpc(self._desc, self._rank)
            logger.info(f'Succeeded to listen on {self._addresses[self._party]}.')
        except Exception as e:
            raise YACLError(
                f'Failed to listen on {self._addresses[self._party]} as exception:\n{e}'
            )

    def send(self, dest_party, data, seq_id):
        msg_bytes = secure_pickle.dumps(
            {'seq_id': seq_id, 'payload': data, 'job': self._job_name}
        )
        self._linker.send_async(self._parties_rank[dest_party], msg_bytes)

        return True

    def recv(self, src_party, seq_id):
        data_log_msg = f"data seq id {seq_id} from {src_party}"
        logger.debug(f"Getting {data_log_msg}")
        rank = self._parties_rank[src_party]

        def _pop_data():
            logger.debug(f"Getted {data_log_msg}.")
            data = self._all_data.pop(seq_id)
            return data

        def _recv_data():
            msg = self._linker.recv(rank)
            msg = secure_pickle.loads(
                msg, filter_type=secure_pickle.FilterType.BLACKLIST
            )
            seq_id = msg['seq_id']
            data = msg['payload']
            job_name = msg['job']
            logger.debug(f'Received data for seq id {seq_id} from {src_party}.')
            if job_name == self._job_name:
                # Avoid bug in unittest, not for production environment.
                # In the unit test, we will repeatedly create and destroy SF clusters.
                # The creation and destruction speeds of different parties are inconsistent.
                # In order to prevent the subsequent cluster receiving data from the previous cluster,
                # ignore data with different name.
                self._all_data[seq_id] = data

        while True:
            if seq_id in self._all_data:
                return _pop_data()
            _recv_data()

    def stop(self):
        if self._linker:
            self._linker.stop_link()
            self._linker = None
