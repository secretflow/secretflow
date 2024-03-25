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

from secretflow.ic.proxy import LinkProxy
from secretflow.ic.remote.call_holder import IcCallHolder
from secretflow.ic.remote.ic_object import IcObject


class IcRemoteClass:
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
        ic_actor_handle = IcActorHandle(
            # _get_addresses(),
            self._cls,
            LinkProxy.self_party,
            self._party,
            self._options,
        )
        ic_call_holder = IcCallHolder(
            self._party, ic_actor_handle.execute_impl, self._options
        )
        ic_call_holder.internal_remote(*cls_args, **cls_kwargs)

        return ic_actor_handle


class IcActorHandle:
    def __init__(
        self,
        # addresses,
        cls,
        party,
        node_party,
        options,
    ) -> None:
        # self._addresses = addresses
        self._body = cls
        self._party = party
        self._node_party = node_party
        self._options = options
        self._actor_handle = None

    def __getattr__(self, method_name: str):
        # User trying to call .bind() without a bind class method
        if method_name == "remote" and "remote" not in dir(self._body):
            raise AttributeError(f".remote() cannot be used again on {type(self)} ")
        # Raise an error if the method is invalid.
        getattr(self._body, method_name)
        call_node = IcActorMethod(
            # self._addresses,
            self._party,
            self._node_party,
            self,
            method_name,
        ).options(**self._options)
        return call_node

    def execute_impl(self, cls_args, cls_kwargs):
        """Executor of ClassNode by ray.remote()

        Args and kwargs are to match base class signature, but not in the
        implementation. All args and kwargs should be resolved and replaced
        with value in bound_args and bound_kwargs via bottom-up recursion when
        current node is executed.
        """
        if self._node_party == self._party:
            self._actor_handle = self._body(*cls_args, **cls_kwargs)

    def execute_remote_method(self, method_name, options, args, kwargs):
        method = getattr(self._actor_handle, method_name)
        return method(*args, **kwargs)


class IcActorMethod:
    def __init__(
        self,
        # addresses,
        party,
        node_party,
        ic_actor_handle,
        method_name,
    ) -> None:
        # self._addresses = addresses
        self._party = party  # Current party
        self._node_party = node_party
        self._ic_actor_handle = ic_actor_handle
        self._method_name = method_name
        self._options = {}
        self._ic_call_holder = IcCallHolder(node_party, self._execute_impl)

    def remote(self, *args, **kwargs) -> IcObject:  # TODO: return type
        return self._ic_call_holder.internal_remote(*args, **kwargs)

    def options(self, **options):
        self._options = options
        self._ic_call_holder.options(**options)
        return self

    def _execute_impl(self, args, kwargs):
        return self._ic_actor_handle.execute_remote_method(
            self._method_name, self._options, args, kwargs
        )
