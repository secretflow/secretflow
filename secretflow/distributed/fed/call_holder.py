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

from jax.tree_util import tree_flatten, tree_unflatten

from .exception import FedLocalError
from .global_context import get_global_context
from .object import FedFuture, FedObject

logger = logging.getLogger(__name__)


class FedCallHolder:
    def __init__(
        self,
        node_party,
        task_name,
        task_impl,
        options={},
    ) -> None:
        self._party = get_global_context().get_party()
        self._node_party = node_party
        self._task_msg = (
            f"TASK:[{task_name}, -ID- {get_global_context().next_task_id()}]"
        )
        self._options = options
        self._task_impl = task_impl

    def options(self, **options):
        self._options = options
        return self

    def recv_dependencies(self, *args, **kwargs):
        flattened_args, _ = tree_flatten((args, kwargs))
        for arg in flattened_args:
            if isinstance(arg, FedObject):
                if arg.get_party() != get_global_context().get_party():
                    logger.debug(
                        f'Try recv {self._task_msg} input '
                        f'seq id {arg.get_seq_id()}, from {arg.get_party()}'
                    )
                    get_global_context().recv(arg)

    def resolve_dependencies(self, *args, **kwargs):
        logger.debug(f'{self._task_msg} wait for inputs')
        flattened_args, tree = tree_flatten((args, kwargs))
        indexes = []
        resolved = []
        for idx, arg in enumerate(flattened_args):
            if isinstance(arg, FedObject):
                indexes.append(idx)
                logger.debug(
                    f'{self._task_msg} wait for input seq id {arg.get_seq_id()}'
                )
                resolved.append(arg.get_object())
                logger.debug(f'{self._task_msg} input seq id {arg.get_seq_id()} ready')
        if resolved:
            for idx, actual_val in zip(indexes, resolved):
                flattened_args[idx] = actual_val

        resolved_args, resolved_kwargs = tree_unflatten(tree, flattened_args)
        logger.debug(f'{self._task_msg} inputs ready')
        return resolved_args, resolved_kwargs

    def internal_remote(self, *args, **kwargs):
        num_returns = 1
        if self._options and 'num_returns' in self._options:
            num_returns = self._options['num_returns']

        if self._party == self._node_party:
            logger.debug(f'Try submit {self._task_msg}')
            self.recv_dependencies(*args, **kwargs)

            def _task():
                resolved = self.resolve_dependencies(*args, **kwargs)
                try:
                    logger.debug(f"{self._task_msg} running")
                    ret = self._task_impl(*resolved)
                    logger.debug(f"{self._task_msg} over")
                    return ret
                except Exception as e:
                    local_err = FedLocalError(e)
                    raise local_err from None

            future = get_global_context().submit_task(_task)
            if num_returns == 1:
                ret = FedObject(
                    get_global_context().get_party(),
                    self._node_party,
                    get_global_context().next_seq_id(),
                    FedFuture(future),
                )
            else:
                ret = [
                    FedObject(
                        get_global_context().get_party(),
                        self._node_party,
                        get_global_context().next_seq_id(),
                        FedFuture(future, num_returns, i),
                    )
                    for i in range(num_returns)
                ]
        else:
            logger.debug(f'Try send {self._task_msg} input')
            flattened_args, _ = tree_flatten((args, kwargs))
            for arg in flattened_args:
                if isinstance(arg, FedObject) and arg.get_party() == self._party:
                    logger.debug(
                        f'Try send {self._task_msg} input '
                        f'seq id {arg.get_seq_id()} to {self._node_party}'
                    )
                    get_global_context().send(self._node_party, arg)
            if num_returns == 1:
                ret = FedObject(
                    get_global_context().get_party(),
                    self._node_party,
                    get_global_context().next_seq_id(),
                )
            else:
                ret = [
                    FedObject(
                        get_global_context().get_party(),
                        self._node_party,
                        get_global_context().next_seq_id(),
                    )
                    for _ in range(num_returns)
                ]

        def _debug_msg():
            if num_returns == 1:
                objs = [ret]
            else:
                objs = ret

            return f"[{', '.join(map(str, objs))}]"

        logger.debug(f"{self._task_msg} generates outputs {_debug_msg()}")
        return ret
