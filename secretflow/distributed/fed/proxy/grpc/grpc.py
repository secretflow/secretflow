# Copyright 2024 Ant Group Co., Ltd.
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

import copy
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import grpc

from secretflow.utils import secure_pickle

from ...exception import FedRemoteError
from ..base import CrossSiloMessageConfig, SenderReceiverProxy
from . import fed_pb2, fed_pb2_grpc

logger = logging.getLogger(__name__)

_GRPC_SERVICE = "SfFedProxy"

_DEFAULT_GRPC_RETRY_POLICY = {
    "maxAttempts": 5,
    "initialBackoff": "5s",
    "maxBackoff": "30s",
    "backoffMultiplier": 2,
    "retryableStatusCodes": ["UNAVAILABLE"],
}


_DEFAULT_GRPC_MAX_SEND_MESSAGE_LENGTH = 500 * 1024 * 1024
_DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 500 * 1024 * 1024

_DEFAULT_GRPC_CHANNEL_OPTIONS = {
    'grpc.enable_retries': 1,
    'grpc.so_reuseport': 0,
    'grpc.max_send_message_length': _DEFAULT_GRPC_MAX_SEND_MESSAGE_LENGTH,
    'grpc.max_receive_message_length': _DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_LENGTH,
    'grpc.service_config': json.dumps(
        {
            'methodConfig': [
                {
                    'name': [{'service': _GRPC_SERVICE}],
                    'retryPolicy': _DEFAULT_GRPC_RETRY_POLICY,
                }
            ]
        }
    ),
}


def get_grpc_options(
    retry_policy=None, max_send_message_length=None, max_receive_message_length=None
):
    if not retry_policy:
        retry_policy = _DEFAULT_GRPC_RETRY_POLICY
    if not max_send_message_length:
        max_send_message_length = _DEFAULT_GRPC_MAX_SEND_MESSAGE_LENGTH
    if not max_receive_message_length:
        max_receive_message_length = _DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_LENGTH

    return [
        (
            'grpc.max_send_message_length',
            max_send_message_length,
        ),
        (
            'grpc.max_receive_message_length',
            max_receive_message_length,
        ),
        ('grpc.enable_retries', 1),
        (
            'grpc.service_config',
            json.dumps(
                {
                    'methodConfig': [
                        {
                            'name': [{'service': _GRPC_SERVICE}],
                            'retryPolicy': retry_policy,
                        }
                    ]
                }
            ),
        ),
        ('grpc.so_reuseport', 0),
    ]


@dataclass
class GrpcCrossSiloMessageConfig(CrossSiloMessageConfig):
    """A class to store parameters used for GRPC communication

    Attributes:
        grpc_retry_policy:
            a dict descibes the retry policy for cross silo rpc call. If None, the
            following default retry policy will be used. More details please refer to
            `retry-policy <https://github.com/grpc/proposal/blob/master/A6-client-retries.md#retry-policy>`_. # noqa

            .. code:: python

                {
                    "maxAttempts": 4,
                    "initialBackoff": "0.1s",
                    "maxBackoff": "1s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": [
                        "UNAVAILABLE"
                    ]
                }
        grpc_channel_options: A list of tuples to store GRPC channel options, e.g.
            .. code:: python

                [
                    ('grpc.enable_retries', 1),
                    ('grpc.max_send_message_length', 50 * 1024 * 1024)
                ]
    """

    grpc_channel_options: Optional[List] = None
    grpc_retry_policy: Optional[Dict[str, str]] = None


def parse_grpc_options(proxy_config: CrossSiloMessageConfig):
    """
    Extract certain fields in `CrossSiloGrpcCommConfig` into the
    "grpc_channel_options". Note that the resulting dict's key
    may not be identical to the config name, but a grpc-supported
    option name.

    Args:
        proxy_config (CrossSiloMessageConfig): The proxy configuration
            from which to extract the gRPC options.

    Returns:
        dict: A dictionary containing the gRPC channel options.
    """
    grpc_channel_options = {}
    if proxy_config is not None:
        # NOTE(NKcqx): `messages_max_size_in_bytes` is a common cross-silo
        # config that should be extracted and filled into proper grpc's
        # channel options.
        # However, `GrpcCrossSiloMessageConfig` provides a more flexible way
        # to configure grpc channel options, i.e. the `grpc_channel_options`
        # field, which may override the `messages_max_size_in_bytes` field.
        if isinstance(proxy_config, CrossSiloMessageConfig):
            if proxy_config.messages_max_size_in_bytes is not None:
                grpc_channel_options.update(
                    {
                        'grpc.max_send_message_length': proxy_config.messages_max_size_in_bytes,
                        'grpc.max_receive_message_length': proxy_config.messages_max_size_in_bytes,
                    }
                )
        if isinstance(proxy_config, GrpcCrossSiloMessageConfig):
            if proxy_config.grpc_channel_options is not None:
                grpc_channel_options.update(proxy_config.grpc_channel_options)
            if proxy_config.grpc_retry_policy is not None:
                grpc_channel_options.update(
                    {
                        'grpc.service_config': json.dumps(
                            {
                                'methodConfig': [
                                    {
                                        'name': [{'service': _GRPC_SERVICE}],
                                        'retryPolicy': proxy_config.grpc_retry_policy,
                                    }
                                ]
                            }
                        ),
                    }
                )

    return grpc_channel_options


def _load_cert_config(cert_config):
    ca_cert, private_key, cert_chain = None, None, None
    if "ca_cert" in cert_config:
        with open(cert_config["ca_cert"], "rb") as file:
            ca_cert = file.read()
    with open(cert_config["key"], "rb") as file:
        private_key = file.read()
    with open(cert_config["cert"], "rb") as file:
        cert_chain = file.read()

    return grpc.ssl_server_credentials(
        [(private_key, cert_chain)],
        root_certificates=ca_cert,
        require_client_auth=ca_cert is not None,
    )


class GrpcProxy(SenderReceiverProxy, fed_pb2_grpc.SfFedProxyServicer):
    def __init__(
        self,
        addresses: Dict,
        party: str,
        job_name: str,
        tls_config: Dict,
        proxy_config: Dict,
    ) -> None:
        proxy_config = GrpcCrossSiloMessageConfig.from_dict(proxy_config)
        super().__init__(job_name, addresses, party, tls_config, proxy_config)
        self._listen_addr = addresses[party]
        grpc_options = copy.deepcopy(_DEFAULT_GRPC_CHANNEL_OPTIONS)
        grpc_options.update(parse_grpc_options(self._proxy_config))
        self._grpc_options = [(k, v) for k, v in grpc_options.items()]
        grpc_metadata = self._proxy_config.http_header or {}
        self._grpc_metadata = [(k, v) for k, v in grpc_metadata.items()]
        self._server = None
        self._stubs: Dict[str, fed_pb2_grpc.SfFedProxyStub] = {}

        self._lock = threading.Lock()
        self._all_data: Dict[int, bytes] = {}
        self._data_events: Dict[int, threading.Event] = {}

    # from grpc base
    def SendData(self, request: fed_pb2.SfFedProxySendData, _):
        job_name = request.job_name
        if job_name != self._job_name:
            logger.warning(
                f"Receive data from job {job_name}, ignore it. "
                f"The reason may be that the ReceiverProxy is listening "
                f"on the same address with that job."
            )
            return fed_pb2.SfFedProxySendDataResponse(
                code=417,
                result=f"JobName mis-match, expected {self._job_name}, got {job_name}.",
            )
        seq_id = request.seq_id
        logger.debug(f'Received a grpc data request seq id {seq_id}')

        with self._lock:
            self._all_data[seq_id] = request.data
            if seq_id not in self._data_events:
                self._data_events[seq_id] = threading.Event()
            event = self._data_events[seq_id]

        event.set()
        logger.debug(f"Event set for seq id {seq_id}")
        return fed_pb2.SfFedProxySendDataResponse(code=200, result="OK")

    # from proxy base
    def concurrent(self):
        return True

    def _start_server(self):
        port = self._listen_addr[self._listen_addr.index(':') + 1 :]
        try:
            logger.info(
                f"ReceiverProxy binding port {port}, options: {self._grpc_options}..."
            )

            server = grpc.server(
                ThreadPoolExecutor(max_workers=8),
                options=self._grpc_options,
            )
            fed_pb2_grpc.add_SfFedProxyServicer_to_server(self, server)

            if self._tls_config:
                server_credentials = _load_cert_config(self._tls_config)
                ## ????? why
                server.add_secure_port(f'[::]:{port}', server_credentials)
            else:
                server.add_insecure_port(f'[::]:{port}')

            server.start()
            logger.info(
                f'Successfully start Grpc service on port {port}, '
                f'with{"" if self._tls_config else "out"} credentials.'
            )
            self._server = server
        except RuntimeError as err:
            logger.error(
                f'Grpc server failed to listen to port: {port}'
                f' Try another port by setting `listen_addr` into `cluster` config'
                f' when calling `fed.init`. Grpc error msg: {err}'
            )
            raise err

    def _init_channel(self):
        grpc_options = self._grpc_options
        for dest_party in self._addresses:
            dest_addr = self._addresses[dest_party]
            if self._tls_config:
                credentials = _load_cert_config(self._tls_config)
                channel = grpc.secure_channel(
                    dest_addr, credentials, options=grpc_options
                )
            else:
                channel = grpc.insecure_channel(dest_addr, options=grpc_options)
            stub = fed_pb2_grpc.SfFedProxyStub(channel)
            self._stubs[dest_party] = stub

    def start(self):
        self._start_server()
        self._init_channel()

    def stop(self):
        if self._server:
            self._server.stop(grace=None).wait()
            self._server = None

    def recv(self, src_party, seq_id):
        data_log_msg = f"data seq_id {seq_id} from {src_party}"
        logger.debug(f"Getting {data_log_msg}")

        with self._lock:
            if seq_id not in self._data_events:
                self._data_events[seq_id] = threading.Event()
            event = self._data_events[seq_id]
        event.wait()
        logger.debug(f"Waited {data_log_msg}.")

        with self._lock:
            data = self._all_data.pop(seq_id)
            self._data_events.pop(seq_id)

        data = secure_pickle.loads(data, filter_type=secure_pickle.FilterType.BLACKLIST)
        if isinstance(data, FedRemoteError):
            logger.error(
                f"Receiving exception: {type(data)}, {data} from {src_party}, "
                f" seq id {seq_id}. Re-raise it."
            )
            from secretflow.distributed.fed.global_context import get_global_context

            get_global_context().set_remote_exception(data)
            raise data
        return data

    def send(self, dest_party, data, seq_id):
        timeout = self._proxy_config.timeout_in_ms / 1000
        data = secure_pickle.dumps(data)
        request = fed_pb2.SfFedProxySendData(
            data=data, seq_id=seq_id, job_name=self._job_name
        )
        response = self._stubs[dest_party].SendData(
            request, metadata=self._grpc_metadata, timeout=timeout
        )
        logger.debug(
            f'Received data response from {dest_party} seq id {seq_id}, '
            f'code: {response.code}, result: {response.result}.'
        )

        if 400 <= response.code < 500:
            logger.warning(
                f"Request was successfully sent but got error response, "
                f"code: {response.code}, message: {response.result}."
            )
            raise RuntimeError(response.result)

        return True
