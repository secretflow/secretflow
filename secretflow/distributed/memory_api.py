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


import inspect
from functools import partial

from secretflow.distributed.memory_actor import MemActorHandle
from secretflow.distributed.memory_function import MemCallHolder


def _is_cython(obj):
    """Check if an object is a Cython function or method"""

    def check_cython(x):
        return type(x).__name__ == "cython_function_or_method"

    # Check if function or method, respectively
    return check_cython(obj) or (
        hasattr(obj, "__func__") and check_cython(obj.__func__)
    )


class MemRemoteFunction:
    def __init__(self, func_or_class) -> None:
        self._node_party = None
        self._func_body = func_or_class
        self._options = {}
        self._mem_call_holder = None

    def party(self, party: str):
        self._node_party = party

        self._mem_call_holder = MemCallHolder(
            self._node_party,
            self._execute_impl,
            self._options,
        )
        return self

    def options(self, **options):
        self._options = options
        if self._mem_call_holder:
            self._mem_call_holder.options(**options)
        return self

    def remote(self, *args, **kwargs):
        if not self.party:
            raise ValueError("You should specify a party name on the remote function.")

        result = self._mem_call_holder.internal_remote(*args, **kwargs)
        return result

    def _execute_impl(self, *args, **kwargs):
        return self._func_body(*args, **kwargs)


class MemRemoteClass:
    def __init__(self, func_or_class) -> None:
        self._party = None
        self._cls = func_or_class
        self._options = {}

    def party(self, party: str):
        self._party = party
        return self

    def options(self, **options):
        self._options = options
        return self

    def remote(self, *cls_args, **cls_kwargs):
        mem_actor_handle = MemActorHandle(
            self._cls,
            self._party,
            self._options,
        )
        mem_actor_handle.execute_impl(cls_args, cls_kwargs)
        return mem_actor_handle


def _make_actor(cls, actor_options):
    return MemRemoteClass(func_or_class=cls).options(**actor_options)


def _make_remote(function_or_class, options):
    if inspect.isfunction(function_or_class) or _is_cython(function_or_class):
        return MemRemoteFunction(function_or_class).options(**options)

    if inspect.isclass(function_or_class):
        return _make_actor(function_or_class, options)

    raise TypeError(
        "The mem.remote decorator must be applied to either a function or a class."
    )


def mem_remote(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # This is the case where the decorator is just @mem.remote.
        # "args[0]" is the class or function under the decorator.
        return _make_remote(args[0], {})
    assert len(args) == 0 and len(kwargs) > 0, "args cannot be none"
    return partial(_make_remote, options=kwargs)
