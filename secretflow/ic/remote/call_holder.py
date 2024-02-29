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

from secretflow.ic.remote import util
from secretflow.ic.remote.ic_object import IcObject
from secretflow.ic.proxy import LinkProxy
from jax.tree_util import tree_flatten


class IcCallHolder:
    """
    `FedCallHolder` represents a call node holder when submitting tasks.
    For example,

    f.party("ALICE").remote()
    ~~~~~~~~~~~~~~~~
        ^
        |
    it's a holder.

    """

    def __init__(
        self,
        node_party: str,
        submit_task_func,
        options={},
    ) -> None:
        self._party = LinkProxy.self_party
        self._node_party = node_party
        self._options = options
        self._submit_task_func = submit_task_func

    def options(self, **options):
        self._options = options
        return self

    def internal_remote(self, *args, **kwargs):
        if not self._node_party:
            raise ValueError("You should specify a party name on the fed actor.")

        # Generate a new fed task id for this call.
        if self._party == self._node_party:
            resolved_args, resolved_kwargs = util.resolve_dependencies(
                self._party, *args, **kwargs
            )
            rets = self._submit_task_func(resolved_args, resolved_kwargs)

            if (
                self._options
                and 'num_returns' in self._options
                and self._options['num_returns'] > 1
            ):
                return [IcObject(self._node_party, ret) for i, ret in enumerate(rets)]
            else:
                return IcObject(self._node_party, rets)
        else:
            flattened_args, _ = tree_flatten((args, kwargs))
            for arg in flattened_args:
                if isinstance(arg, IcObject) and arg.get_party() == self._party:
                    if not arg.was_sending_or_sent_to_party(self._node_party):
                        arg.mark_is_sending_to_party(self._node_party)
                        LinkProxy.send(
                            dest_party=self._node_party,
                            data=arg.data,
                        )

            if (
                self._options
                and 'num_returns' in self._options
                and self._options['num_returns'] > 1
            ):
                num_returns = self._options['num_returns']
                return [IcObject(self._node_party, None) for i in range(num_returns)]
            else:
                return IcObject(self._node_party, None)
