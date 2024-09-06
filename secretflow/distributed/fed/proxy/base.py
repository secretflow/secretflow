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
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class CrossSiloMessageConfig:
    """A class to store parameters used for Proxy Actor.

    Attributes:
        proxy_max_restarts:
            The max restart times for the send proxy.
        send_resource_label:
            Customized resource label, the SenderProxyActor
            will be scheduled based on the declared resource label. For example,
            when setting to `{"my_label": 1}`, then the sender proxy actor will be
            started only on nodes with `{"resource": {"my_label": $NUM}}` where
            $NUM >= 1.
        recv_resource_label:
            Customized resource label, the ReceiverProxyActor
            will be scheduled based on the declared resource label. For example,
            when setting to `{"my_label": 1}`, then the receiver proxy actor will be
            started only on nodes with `{"resource": {"my_label": $NUM}}` where
            $NUM >= 1.
        exit_on_sending_failure:
            whether exit when failure on cross-silo sending. If True, a SIGINT will be
            signaled to self if failed to sending cross-silo data and exit then.
        continue_waiting_for_data_sending_on_error:
            Whether to continue waiting for data sending if an error occurs, including
            data-sending errors and receiving errors from the peer. If True, wait until
            all data has been sent.
        messages_max_size_in_bytes:
            The maximum length in bytes of cross-silo messages. If None, the default
            value of 500 MB is specified.
        timeout_in_ms:
            The timeout in mili-seconds of a cross-silo RPC call. It's 60000 by
            default.
        http_header:
            The HTTP header, e.g. metadata in grpc, sent with the RPC request.
            This won't override basic tcp headers, such as `user-agent`, but concat
            them together.
        max_concurrency:
            the max_concurrency of the sender/receiver proxy actor.
        use_global_proxy:
            Whether using the global proxy actor or create new proxy actor for current
            job.
    """

    proxy_max_restarts: Optional[int] = None
    timeout_in_ms: Optional[int] = 60000
    messages_max_size_in_bytes: Optional[int] = None
    exit_on_sending_failure: Optional[bool] = False
    continue_waiting_for_data_sending_on_error: Optional[bool] = False
    send_resource_label: Optional[Dict[str, str]] = None
    recv_resource_label: Optional[Dict[str, str]] = None
    http_header: Optional[Dict[str, str]] = None
    max_concurrency: Optional[int] = None
    expose_error_trace: Optional[bool] = False
    use_global_proxy: Optional[bool] = True

    def __json__(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict) -> CrossSiloMessageConfig:
        """Initialize CrossSiloMessageConfig from a dictionary.

        Args:
            data (Dict): Dictionary with keys as member variable names.

        Returns:
            CrossSiloMessageConfig: An instance of CrossSiloMessageConfig.
        """
        # Get the attributes of the class
        data = data or {}
        attrs = [field.name for field in fields(cls)]
        # Filter the dictionary to only include keys that are attributes of the class
        filtered_data = {key: value for key, value in data.items() if key in attrs}
        return cls(**filtered_data)


class SenderReceiverProxy(abc.ABC):
    def __init__(
        self,
        job_name: str,
        addresses: Dict,
        self_party: str,
        tls_config: Dict,
        proxy_config: CrossSiloMessageConfig = None,
    ) -> None:
        self._job_name = job_name
        self._addresses = addresses
        self._party = self_party
        self._tls_config = tls_config
        self._proxy_config = proxy_config

    @abc.abstractmethod
    def start(self) -> None:
        """Start recv proxy service, block until service is ready"""
        pass

    @abc.abstractmethod
    def recv(self, src_party: str, seq_id: Optional[int]) -> Any:
        """Recv data with seq_id from src_party"""
        pass

    @abc.abstractmethod
    def send(self, dest_party: str, data: Any, seq_id: Optional[int]) -> bool:
        """Send data with seq_id to dest_party"""
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop recv proxy service"""
        pass

    @abc.abstractmethod
    def concurrent(self) -> bool:
        """If this proxy can be called concurrently"""
        pass
