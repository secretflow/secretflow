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
from abc import ABC, abstractmethod
from threading import Condition
from typing import Any, Dict, List, Union

import fed
import ray

import secretflow.distributed as sfd
from secretflow.device.driver import reveal
from secretflow.distributed.primitive import DISTRIBUTION_MODE

from .device import PYU

thread_local = threading.local()

SERVER = "server"
CLIENT = "client"


def get_role():
    return thread_local.link.role


def get_device():
    return thread_local.link.device


def set_mesh(link: 'Link'):
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


class Communicator(ABC):
    @abstractmethod
    def send(dest: PYU, data: Any, key: str):
        raise NotImplementedError()

    @abstractmethod
    def recv(src: PYU, keys: Union[str, List[str]]):
        raise NotImplementedError()


class FedCommunicator(Communicator):
    def __init__(self, partners: List[PYU]):
        self.parties = [partner.party for partner in partners]

    def send(self, dest: PYU, data: Any, key: str):
        assert dest.party in self.parties, f'Device {dest} is not in this communicator.'
        return fed.send(
            dest_party=dest.party,
            data=data,
            upstream_seq_id=key,
            downstream_seq_id=key,
        )

    def recv(self, src: PYU, keys: Union[str, List[str]]):
        is_single = isinstance(keys, str)
        if is_single:
            keys = [keys]

        vals = ray.get([fed.recv(src.party, src.party, key, key) for key in keys])
        return vals[0] if is_single else vals


class RayCommunicator(Communicator):
    def __init__(self):
        self._messages = {}
        self._cv = Condition()

    def links(self, links: Dict[PYU, ray.actor.ActorHandle]):
        self._links = links

    def send(self, dest: PYU, data: Any, key: str):
        assert dest in self._links, f'Device {dest} is not in this communicator.'
        logging.debug(f'send to dest {dest}')
        self._links[dest]._recv_message.remote(key, data)

    def _recv_message(self, key: str, value: Any):
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

    def recv(self, src: PYU, keys: Union[str, List[str]]):
        logging.debug(f'receive message: {keys}')

        is_single = isinstance(keys, str)
        if is_single:
            keys = [keys]
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

        return list(vals.values())[0] if is_single else list(vals.values())


class Link:
    """A helper class for communication inside actor between several actors.

    You should not use this class directly but inherit it and decorate your
    child class with :py:meth:`~secretflow.device.proxy`.

    Examples
    --------
    >>> from secretflow.device import proxy
    >>> from seceretflow.device.link import Link, init_link
    >>>
    >>> @proxy
    >>> class PS(Link):
    >>>    def run(self):
    >>>        pass
    >>>
    >>>
    >>> @proxy
    >>> class Client(Link):
    >>>    def run(self):
    >>>        pass
    >>>
    >>> ps = PS()
    >>> clients = [Client() for i in range(2)]
    >>> init_link(ps, clients)
    >>> for client in clients:
    >>>     init_link(client, ps)
    >>>
    """

    def __init__(self, device: PYU, key_prefix: str = ''):
        """Initialize

        Args:
            device: where this Link instance located, PYU
        """
        self._device = device
        self._initialized = False
        self._clients = None
        self._server = None
        self._key_prefix = key_prefix
        self._comm = None

    def initialize(
        self, comm_or_links: Union[Communicator, Dict[PYU, ray.actor.ActorHandle]]
    ):
        if isinstance(comm_or_links, FedCommunicator):
            self._comm = comm_or_links
        else:
            self._comm = RayCommunicator()
            self._comm.links(comm_or_links)
        # Indicate success.
        return True

    @staticmethod
    def _create_key(
        src_device: Union[PYU, List[PYU]],
        dst_device: Union[PYU, List[PYU]],
        name: str,
        step_id: int = 0,
        key_prefix='',
    ):
        if isinstance(src_device, PYU) and isinstance(dst_device, PYU):
            return f'{key_prefix};{src_device};{dst_device};{name};{step_id}'
        elif isinstance(src_device, List):
            assert isinstance(dst_device, PYU), f'invalid dst_device: {dst_device}'
            return [
                f'{key_prefix};{device};{dst_device};{name};{step_id}'
                for device in src_device
            ]
        elif isinstance(dst_device, List):
            assert isinstance(src_device, PYU), f'invalid src_device: {src_device}'
            return [
                f'{key_prefix};{src_device};{device};{name};{step_id}'
                for device in dst_device
            ]
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

        key = self._create_key(
            self._device, dst_device, name, step_id, self._key_prefix
        )
        logging.debug(f'send message: {key}')

        if isinstance(dst_device, list):
            for msg_id, device in zip(key, dst_device):
                self._comm.send(device, value, msg_id)
        else:
            self._comm.send(dst_device, value, key)

    def _recv_message(self, key: str, value: Any):
        """Receive message

        Args:
            key: The message key, consisting by source & destination device,
                message name, and unique identifier
            value: message body
        """
        logging.debug(f'receive message from remote: {key}')
        self._comm._recv_message(key, value)

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

        key = self._create_key(
            src_device, self._device, name, step_id, self._key_prefix
        )
        logging.debug(f'receive message: {key}')
        return self._comm.recv(src=self._device, keys=key)


def init_link(link: Link, partners: List[Link]):
    if not isinstance(partners, list):
        partners = [partners]
    if sfd.get_distribution_mode() == DISTRIBUTION_MODE.PRODUCTION:
        comm = FedCommunicator([partner.device for partner in partners])
        # Use `get` here as a barrier to make sure that initialize is done at first.
        # Note that link should be a `proxy`ed actor.
        reveal(link.initialize(comm))
    else:
        reveal(link.initialize({partner.device: partner.data for partner in partners}))
