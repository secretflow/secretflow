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
from secretflow.ic.remote.ic_object import IcObject
from jax.tree_util import tree_flatten, tree_unflatten


def resolve_dependencies(current_party, *args, **kwargs):
    flattened_args, tree = tree_flatten((args, kwargs))
    indexes = []
    resolved = []
    for idx, arg in enumerate(flattened_args):
        if isinstance(arg, IcObject):
            indexes.append(idx)
            if arg.get_party() == current_party:
                resolved.append(arg.data)
            else:
                if not arg.received:
                    arg.data = LinkProxy.recv(arg.get_party())
                    arg.mark_received()
                resolved.append(arg.data)
    if resolved:
        for idx, actual_val in zip(indexes, resolved):
            flattened_args[idx] = actual_val

    resolved_args, resolved_kwargs = tree_unflatten(tree, flattened_args)
    return resolved_args, resolved_kwargs
