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

import logging
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class FedFuture:
    future: Future
    num_returns: int = 1
    obj_index: int = 0

    def get(self) -> Any:
        while True:
            try:
                future_result = self.future.result(timeout=2)
                break
            except TimeoutError:
                pass

            # check global exception
            from secretflow.distributed.fed.global_context import get_global_context

            exception = get_global_context().get_last_exception()
            if exception:
                logging.warning(f"get obj interrupted by {exception}")
                raise exception

        if self.num_returns == 1:
            return future_result
        else:
            return future_result[self.obj_index]


class FedObject:
    def __init__(
        self, current_party: str, obj_node_party: str, seq_id: int, obj: Any = None
    ) -> None:
        # current_party is the party of the current runtime context
        self._current_party = current_party
        # obj_node_party is the party produced this object
        self._obj_node_party = obj_node_party
        # A FedObject is a reference to a specific output variable of a specific function in a specific call.
        # This seq_id is a unique id in fed runtime context to mark this output variable.
        # And seq_id is finalized when 'driver code' called function's remote() in main thread.
        # for details, see FedCallHolder::internal_remote
        self._seq_id = seq_id

        self._lock = Lock()
        self._object = obj

        if self._obj_node_party == self._current_party:
            # if object is produced by current_party
            # the reference to the object will set during __init__
            self._has_object = True
            # is_sent is meaningless if object is not produced by current_part
            # and this sent flag is used inside sf.fed, not a public api,
            # can only be changed by GlobalContext.
            # for details, see GlobalContext::send
            self._is_sent = set()
        else:
            # Otherwise, FedObject don't has object before recv object from obj_node_party
            self._has_object = False

    def __str__(self) -> str:
        ret = f"FedObj seq id {self._seq_id} on {self._obj_node_party} "
        if self._has_object:
            ret += f"with object {id(self._object)} type {type(self._object)}"
        else:
            ret += "without object"
        return ret

    def __deepcopy__(self, memodict={}):
        return self

    def get_party(self) -> str:
        return self._obj_node_party

    def get_seq_id(self) -> str:
        return self._seq_id

    def mark_send(self, target_party: str) -> None:
        if self._obj_node_party != self._current_party:
            raise RuntimeError(
                "is_sent is meaningless if object is not produced by current_party."
                "Do not use this flag in this case."
            )
        self._is_sent.add(target_party)

    def is_sent(self, target_party: str) -> bool:
        if self._obj_node_party != self._current_party:
            raise RuntimeError(
                "is_sent is meaningless if object is not produced by current_party."
                "Do not use this flag in this case."
            )
        return target_party in self._is_sent

    def get_object(self) -> Any:
        assert self._has_object
        with self._lock:
            if isinstance(self._object, FedFuture):
                self._object = self._object.get()
            return self._object

    def has_object(self) -> bool:
        return self._has_object

    def set_object(self, obj: FedFuture) -> None:
        assert self._obj_node_party != self._current_party
        assert isinstance(obj, FedFuture)
        self._has_object = True
        with self._lock:
            self._object = obj
