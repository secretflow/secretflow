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

import abc
import logging
from typing import Tuple, List
from secretflow.ic.proto.handshake.entry_pb2 import HandshakeRequest, HandshakeResponse
from secretflow.ic.proto.common.header_pb2 import OK
from secretflow.ic.proxy import LinkProxy


class IcHandler(abc.ABC):
    def __init__(self, dataset):
        self._dataset = dataset

    def run(self):
        if not self._handshake():
            logging.warning('interconnection handshake failed')
            return None
        self._run_algo()

    def _handshake(self) -> bool:
        if LinkProxy.self_party == LinkProxy.all_parties[0]:
            print('++++++++++++ passive handshake +++++++++++++++')
            return self._passive_handshake()
        else:
            print('++++++++++++ active handshake +++++++++++++++')
            return self._active_handshake()

    def _active_handshake(self) -> bool:
        request = self._build_handshake_request()
        self._send_handshake_request(request)
        response = self._recv_handshake_response()
        return self._process_handshake_response(response)

    def _passive_handshake(self) -> bool:
        requests = self._recv_handshake_requests()
        response = self._process_handshake_requests(requests)
        self._send_handshake_response(response)
        return response.header.error_code == OK

    @abc.abstractmethod
    def _build_handshake_request(self) -> HandshakeRequest:
        raise NotImplementedError()

    @abc.abstractmethod
    def _build_handshake_response(self) -> HandshakeResponse:
        response = HandshakeResponse()
        response.header.error_code = OK
        return response

    @staticmethod
    def _send_handshake_request(request: HandshakeRequest):
        LinkProxy.send_raw(LinkProxy.all_parties[0], request.SerializeToString())
        logging.info(f'send handshake request: {request}')

    @staticmethod
    def _recv_handshake_requests() -> List[HandshakeRequest]:
        requests = []
        for party in LinkProxy.all_parties:
            if party == LinkProxy.self_party:
                continue
            msg_bytes = LinkProxy.recv_raw(src_party=party)
            request = HandshakeRequest()
            request.ParseFromString(msg_bytes)
            requests.append(request)
            logging.info(f'recv handshake request from {party}: {request}')

        return requests

    @staticmethod
    def _send_handshake_response(response: HandshakeResponse):
        logging.info(f'send response: {response}')
        for party in LinkProxy.all_parties:
            if party == LinkProxy.self_party:
                continue
            LinkProxy.send_raw(party, response.SerializeToString())

    @staticmethod
    def _recv_handshake_response() -> HandshakeResponse:
        msg_bytes = LinkProxy.recv_raw(LinkProxy.all_parties[0])
        response = HandshakeResponse()
        response.ParseFromString(msg_bytes)
        logging.info(f'recv handshake response: {response}')
        return response

    def _process_handshake_requests(
        self, requests: List[HandshakeRequest]
    ) -> HandshakeResponse:
        err_code, err_msg = self._negotiate_handshake_params(requests)
        if err_code != OK:
            response = HandshakeResponse()
            response.header.error_code = err_code
            response.header.error_msg = err_msg
            return response

        return self._build_handshake_response()

    @abc.abstractmethod
    def _negotiate_handshake_params(
        self, requests: List[HandshakeRequest]
    ) -> Tuple[int, str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _process_handshake_response(self, response: HandshakeResponse) -> bool:
        return response.header.error_code == OK

    @abc.abstractmethod
    def _run_algo(self):
        raise NotImplementedError()
