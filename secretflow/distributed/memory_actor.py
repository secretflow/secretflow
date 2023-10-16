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

from secretflow.distributed.memory_function import MemCallHolder


class MemActorHandle:
    def __init__(
        self,
        cls,
        node_party,
        options,
    ) -> None:
        self._body = cls
        self._node_party = node_party
        self._options = options
        self._actor_handle = None

    def __getattr__(self, method_name: str):
        # User trying to call .bind() without a bind class method
        if method_name == "remote" and "remote" not in dir(self._body):
            raise AttributeError(f".remote() cannot be used again on {type(self)} ")
        # Raise an error if the method is invalid.
        getattr(self._body, method_name)
        method_actor = MemActorMethod(
            self._node_party,
            self,
            method_name,
        ).options(**self._options)
        return method_actor

    def execute_impl(self, cls_args, cls_kwargs):
        """Executor of ClassNode by mem.remote()

        Args and kwargs are to match base class signature, but not in the
        implementation. All args and kwargs should be resolved and replaced
        with value in bound_args and bound_kwargs via bottom-up recursion when
        current node is executed.
        """
        self._actor_handle = self._body(*cls_args, **cls_kwargs)

    def _execute_method(self, method_name, options, *args, **kwargs):
        num_returns = 1
        if options and 'num_returns' in options:
            num_returns = options['num_returns']
        logging.debug(f"Actor method call: {method_name}, num_returns: {num_returns}")
        # execute method call
        function = getattr(self._actor_handle, method_name)

        return function(*args, **kwargs)


class MemActorMethod:
    def __init__(
        self,
        node_party,
        mem_actor_handle,
        method_name,
    ) -> None:
        self._node_party = node_party
        self._mem_actor_handle = mem_actor_handle
        self._method_name = method_name
        self._options = {}
        self._mem_call_holder = MemCallHolder(node_party, self._execute_impl)

    def remote(self, *args, **kwargs) -> object:
        return self._mem_call_holder.internal_remote(*args, **kwargs)

    def options(self, **options):
        self._options = options
        self._mem_call_holder.options(**options)
        return self

    def _execute_impl(self, *args, **kwargs):
        return self._mem_actor_handle._execute_method(
            self._method_name, self._options, *args, **kwargs
        )
