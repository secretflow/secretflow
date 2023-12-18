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
import functools
from typing import Union, List
from secretflow.ic.proxy import LinkProxy
from secretflow.ic.remote.remote_function import IcRemoteFunction
from secretflow.ic.remote.remote_class import IcRemoteClass
from secretflow.ic.remote.ic_object import IcObject


def remote(*args, **kwargs):
    def _make_ic_remote(function_or_class, **options):
        if inspect.isfunction(function_or_class) or _is_cython(function_or_class):
            return IcRemoteFunction(function_or_class).options(**options)

        if inspect.isclass(function_or_class):
            return IcRemoteClass(function_or_class).options(**options)

        raise TypeError(
            "The @ic.remote decorator must be applied to either a function or a class."
        )

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # This is the case where the decorator is just @ray.remote.
        # "args[0]" is the class or function under the decorator.
        return _make_ic_remote(args[0])
    assert len(args) == 0 and len(kwargs) > 0, "Remote args error."
    return functools.partial(_make_ic_remote, options=kwargs)


def _is_cython(obj):
    """Check if an object is a Cython function or method"""

    # TODO(suo): We could split these into two functions, one for Cython
    # functions and another for Cython methods.
    # TODO(suo): There doesn't appear to be a Cython function 'type' we can
    # check against via isinstance. Please correct me if I'm wrong.
    def check_cython(x):
        return type(x).__name__ == "cython_function_or_method"

    # Check if function or method, respectively
    return check_cython(obj) or (
        hasattr(obj, "__func__") and check_cython(obj.__func__)
    )


def get(ic_objects: Union[IcObject, List[IcObject]]):
    is_individual_id = isinstance(ic_objects, IcObject)
    if is_individual_id:
        ic_objects = [ic_objects]

    values = []
    for ic_object in ic_objects:
        if ic_object.get_party() == LinkProxy.self_party:
            # assert ic_object.data is not None
            values.append(ic_object.data)

            for party in LinkProxy.all_parties:
                if (
                    party != LinkProxy.self_party
                    and not ic_object.was_sending_or_sent_to_party(party)
                ):
                    ic_object.mark_is_sending_to_party(party)
                    LinkProxy.send(dest_party=party, data=ic_object.data)
        else:
            if not ic_object.received:
                ic_object.data = LinkProxy.recv(ic_object.get_party())
                ic_object.mark_received()
            values.append(ic_object.data)

    if is_individual_id:
        values = values[0]

    return values
