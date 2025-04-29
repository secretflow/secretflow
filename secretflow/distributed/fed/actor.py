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
from functools import partial
from threading import Event, Lock
from typing import Dict

from .call_holder import FedCallHolder
from .exception import main_thread_assert

logger = logging.getLogger(__name__)


class FedActorHandle:
    def __init__(
        self,
        cls,
        party,
        node_party,
        options,
    ) -> None:
        self._cls = cls
        self._party = party
        self._node_party = node_party
        self._options = options
        self._cls_fed_obj = None
        self._max_concurrency = self._options.get("max_concurrency", None)
        if self._max_concurrency is None:
            self._max_concurrency = 1
        assert isinstance(self._max_concurrency, int) and self._max_concurrency > 0
        self._method_call_seq_id = 0
        self._lock = Lock()
        self._events: Dict[int, Event] = {}
        self._readable_name = f"FedActor({id(self)})[{self._cls.__name__}]"
        logger.debug(f"FedActorHandle create {self._readable_name}")

    def _execute_impl(self, cls_args, cls_kwargs):
        return self._cls(*cls_args, **cls_kwargs)

    def _method_seq_wait(self, seq_id, method_name):
        logger.debug(
            f"{self._readable_name}::{method_name}@{seq_id} waiting concurrency"
        )
        wait_on = seq_id - self._max_concurrency
        if wait_on > 0:
            with self._lock:
                event = self._events[wait_on]
            event.wait()
            with self._lock:
                del self._events[wait_on]
        logger.debug(f"{self._readable_name}::{method_name}@{seq_id} ready to running")

    def _method_notify(self, seq_id, method_name):
        logger.debug(f"{self._readable_name}::{method_name}@{seq_id} over")
        with self._lock:
            event = self._events[seq_id]
        event.set()

    def remote(self, *args, **kwargs):
        main_thread_assert()
        fed_call_holder = FedCallHolder(
            self._node_party,
            f"{self._cls.__name__}::__init__",
            self._execute_impl,
            self._options,
        )
        self._cls_fed_obj = fed_call_holder.internal_remote(*args, **kwargs)

    def __getattr__(self, method_name: str):
        main_thread_assert()
        if method_name == "remote" and "remote" not in dir(self._cls):
            raise AttributeError(f".remote() cannot be used again on {self._cls} ")
        # Raise an error if the method is invalid.
        getattr(self._cls, method_name)

        if self._party == self._node_party:
            self._method_call_seq_id += 1
            with self._lock:
                self._events[self._method_call_seq_id] = Event()
            return FedActorMethod(
                self._node_party,
                self._cls_fed_obj,
                partial(self._method_seq_wait, self._method_call_seq_id, method_name),
                partial(self._method_notify, self._method_call_seq_id, method_name),
                method_name,
                self._cls.__name__,
            ).options(**self._options)
        else:
            return FedActorMethod(
                self._node_party,
                None,
                None,
                None,
                method_name,
                self._cls.__name__,
            ).options(**self._options)


class FedActorMethod:
    def __init__(
        self,
        node_party,
        cls_fed_obj,
        wait_on,
        notify,
        method_name,
        cls_name,
    ) -> None:
        self._node_party = node_party
        self._cls_fed_obj = cls_fed_obj
        self._wait_on = wait_on
        self._notify = notify
        self._method_name = method_name
        self._options = {}
        full_name = f"{cls_name}::{method_name}"
        self._fed_call_holder = FedCallHolder(node_party, full_name, self._execute_impl)
        logger.debug(f"FedActorMethod create {full_name} id {id(self)}")

    def remote(self, *args, **kwargs):
        main_thread_assert()
        return self._fed_call_holder.internal_remote(*args, **kwargs)

    def options(self, **options):
        main_thread_assert()
        self._options = options
        self._fed_call_holder.options(**options)
        return self

    def _execute_impl(self, args, kwargs):
        try:
            cls_instance = self._cls_fed_obj.get_object()
            func = getattr(cls_instance, self._method_name)
            self._wait_on()
            ret = func(*args, **kwargs)
        finally:
            self._notify()
        return ret
