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
import threading
from threading import Condition
from typing import Any, Dict, List, Union

from .device import PYU, Device

thread_local = threading.local()

SERVER = "server"
CLIENT = "client"


def get_role():
    return thread_local.link.role


def get_device():
    return thread_local.link.device


def set_mesh(link):
    thread_local.link = link


def send_to_clients(name, value, version):
    """Send message to the target device.
        this function is non-blocking.

    Args:
        name: message name
        value: message value
        version: message version, used to distinguish between different training rounds
    """
    thread_local.link.send(name, value, thread_local.link._clients, version)


def send_to_server(name, value, version):
    """Send message to the target device.
        this function is non-blocking.

    Args:
        name: message name
        value: message value
        version: message version, used to distinguish between different training rounds
    """
    thread_local.link.send(name, value, thread_local.link._server, version)


def recv_from_clients(name, version):
    """
    Receive messages from the source device.
        this function is blocking

    Args:
        name: message name
        version: TODO: What is the purpose of the version parameter?

    Returns:
        The received message
    """
    return thread_local.link.recv(name, thread_local.link._clients, version)


def recv_from_server(name, version):
    """
    Receive messages from the source device.
        this function is blocking

    Args:
        name: message name
        version: message version, used to distinguish between different training rounds

    Returns:
        The received message
    """
    return thread_local.link.recv(name, thread_local.link._server, version)


class Link:
    def __init__(self, device: PYU):
        """Initialize

        Args:
            device: where this Link instance located, PYU
        """
        self._device = device
        self._comm: Dict[Device, 'Link'] = {}
        self._initialized = False
        self._messages = {}
        self._cv = Condition()
        self._clients = None
        self._server = None

    def initialize(self, comm: Dict[Device, 'Link']):
        """Initialize networking

        Args:
            comm: A dict of {target device, communication (link) object}
        """
        assert not self._initialized, f're-initialize is not allowed'
        self._comm = comm
        self._initialized = True

    @staticmethod
    def __create_key(
        src_device: Union[PYU, List[PYU]],
        dst_device: Union[PYU, List[PYU]],
        name: str,
        step_id: int = 0,
    ):
        if isinstance(src_device, PYU) and isinstance(dst_device, PYU):
            return f'{src_device};{dst_device};{name};{step_id}'
        elif isinstance(src_device, List):
            assert isinstance(dst_device, PYU), f'invalid dst_device: {dst_device}'
            return [f'{device};{dst_device};{name};{step_id}' for device in src_device]
        elif isinstance(dst_device, List):
            assert isinstance(src_device, PYU), f'invalid src_device: {src_device}'
            return [f'{src_device};{device};{name};{step_id}' for device in dst_device]
        else:
            assert False, f'invalid src_device: {src_device}, dst_device: {dst_device}'

    @property
    def clients(self):
        return self._clients

    @clients.setter
    def clients(self, clients: List[PYU]):
        self._clients = clients

    @property
    def server(self):
        return self._server

    @server.setter
    def server(self, server: PYU):
        self._server = server

    def send(
        self, name: str, value: Any, dst_device: Union[PYU, List[PYU]], step_id: int = 0
    ):
        """Send message to target device.
            this function is non-blocking

        Args:
            name: message name
            value: message value
            dst_device: target device(s), can be a single device or a list of devices
            step_id: A process-level unique identifier to identify the communication
        """
        assert isinstance(dst_device, PYU) or (
            isinstance(dst_device, List) and len(dst_device) > 0
        ), f'dst_device must be PYU or PYU list'

        key = self.__create_key(self._device, dst_device, name, step_id)
        logging.debug(f'send message: {key}')

        if isinstance(key, str):
            self._comm[dst_device].recv_message.remote(key, value)
        else:
            for k, device in zip(key, dst_device):
                self._comm[device].recv_message.remote(k, value)

    def recv_message(self, key: str, value: Any):
        """Receive message

        Args:
            key: The message key, consisting by source & destination device,
                message name, and unique identifier
            value: message body
        """
        logging.debug(f'receive message from remote: {key}')
        with self._cv:
            self._messages[key] = value
            self._cv.notifyAll()

    def recv(
        self, name: str, src_device: Union[PYU, List[PYU]], step_id: int = 0
    ) -> Any:
        """Receive messages from the source device.
            this function is blocking

        Args:
            name: The message name
            src_device: source device(s), can be a single device or a list of devices
            step_id: A process-level unique identifier to identify the communication

        Returns:
            The received message
        """
        assert isinstance(src_device, PYU) or (
            isinstance(src_device, List) and len(src_device) > 0
        ), f'dst_device must be PYU or PYU list'

        key = self.__create_key(src_device, self._device, name, step_id)
        logging.debug(f'receive message: {key}')

        keys = {key} if isinstance(key, str) else set(key)
        vals = {}
        with self._cv:
            while True:
                recv_keys = []
                for k in keys:
                    if k in self._messages:
                        vals[k] = self._messages.pop(k)
                        recv_keys.append(k)

                for k in recv_keys:
                    keys.remove(k)

                if len(keys) == 0:
                    break

                self._cv.wait()

        return vals[key] if isinstance(key, str) else [vals[k] for k in key]
