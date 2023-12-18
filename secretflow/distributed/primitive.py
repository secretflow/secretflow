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
import logging
from enum import Enum, unique
from functools import partial
from typing import List, Union

import fed
import jax
import ray
import secretflow.ic.remote as ic
from ray import Language
from ray._private import ray_option_utils
from ray.actor import ActorClass, _inject_tracing_into_class, ray_constants
from ray.remote_function import RemoteFunction

from secretflow.distributed.memory_api import mem_remote
from secretflow.utils.ray_compatibility import ray_version_less_than_2_0_0


@unique
class DISTRIBUTION_MODE(Enum):
    PRODUCTION = 1
    SIMULATION = 2
    DEBUG = 3
    INTERCONNECTION = 4


_distribution_mode = DISTRIBUTION_MODE.SIMULATION
_is_cluster_active = False
_current_cluster_id = -1


def set_distribution_mode(mode: DISTRIBUTION_MODE):
    global _distribution_mode
    if mode in DISTRIBUTION_MODE:
        _distribution_mode = mode
    else:
        raise Exception(f"Illegal distribute mode, only support ({DISTRIBUTION_MODE})")


def get_distribution_mode():
    global _distribution_mode
    return _distribution_mode


def get_current_cluster_idx():
    """
    Get the current secretflow cluster index.
    In general situation, users will execute sf.init() at the first of their codes,
    and execute sf.shutdown() at the end.
    But in some cases, the following code may appear:
        - sf.init()
        - do_something_in_sf()
        - sf.shutdown()
        - do_something_else()
        - sf.init()
        - do_something_in_sf()
        - sf.shutdown()
    To ensure some variable can work as expect when it across different clusters,
    we use an index to indicate diferent active cluster.
    As shown above, between the first sf.init() and sf.shutdown(), the cluster index is 0,
    and the between the second sf.init() and sf.shutdown(), the cluster index is 1,
    In do_something_else(), the cluster index is -1 (no active cluster).

    Returns: The current cluster id, -1 for not active.

    """
    global _current_cluster_id
    global _is_cluster_active
    if _is_cluster_active:
        return _current_cluster_id
    else:
        return -1


def active_sf_cluster():
    """
    Record the cluster's active status and generate the current cluster index.
    """
    global _current_cluster_id
    global _is_cluster_active
    _current_cluster_id = _current_cluster_id + 1
    _is_cluster_active = True


def get_cluster_avaliable_resources():
    if get_distribution_mode() == DISTRIBUTION_MODE.DEBUG:
        raise NotImplementedError(
            "Need but not implement. This will be used by tune when debug mode."
        )
    elif get_distribution_mode() == DISTRIBUTION_MODE.SIMULATION:
        return ray.available_resources()
    else:
        raise NotImplementedError(
            "Need but not implement. This will be used by tune when production mode."
        )


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


def remote(*args, **kwargs):
    if get_distribution_mode() == DISTRIBUTION_MODE.PRODUCTION:
        return fed.remote(*args, **kwargs)
    elif get_distribution_mode() == DISTRIBUTION_MODE.SIMULATION:
        return ray_remote(*args, **kwargs)
    elif get_distribution_mode() == DISTRIBUTION_MODE.DEBUG:
        return mem_remote(*args, **kwargs)
    elif get_distribution_mode() == DISTRIBUTION_MODE.INTERCONNECTION:
        return ic.remote(*args, **kwargs)
    else:
        raise Exception(f"Illegal distribute mode, only support ({DISTRIBUTION_MODE})")


def get(
    object_refs: Union[
        Union[ray.ObjectRef, List[ray.ObjectRef]],
        Union[fed.FedObject, List[fed.FedObject]],
        Union[object, List[object]],
    ]
):
    if get_distribution_mode() == DISTRIBUTION_MODE.PRODUCTION:
        return fed.get(object_refs)
    elif get_distribution_mode() == DISTRIBUTION_MODE.SIMULATION:
        return ray.get(object_refs)
    elif get_distribution_mode() == DISTRIBUTION_MODE.DEBUG:
        return object_refs
    elif get_distribution_mode() == DISTRIBUTION_MODE.INTERCONNECTION:
        return ic.get(object_refs)
    else:
        raise Exception(f"Illegal distribute mode, only support ({DISTRIBUTION_MODE})")


def kill(actor, *, no_restart=True):
    if get_distribution_mode() == DISTRIBUTION_MODE.PRODUCTION:
        return fed.kill(actor, no_restart=no_restart)
    elif get_distribution_mode() == DISTRIBUTION_MODE.SIMULATION:
        return ray.kill(actor, no_restart=no_restart)
    elif get_distribution_mode() == DISTRIBUTION_MODE.DEBUG:
        logging.warning("Actor be killed")
    elif get_distribution_mode() == DISTRIBUTION_MODE.INTERCONNECTION:
        logging.warning("Actor be killed")
    else:
        raise Exception(f"Illegal distribute mode, only support ({DISTRIBUTION_MODE})")


def shutdown():
    global _is_cluster_active
    _is_cluster_active = False
    if get_distribution_mode() == DISTRIBUTION_MODE.PRODUCTION:
        fed.shutdown()
        ray.shutdown()
    elif get_distribution_mode() == DISTRIBUTION_MODE.SIMULATION:
        ray.shutdown()
    elif get_distribution_mode() == DISTRIBUTION_MODE.DEBUG:
        logging.warning("Shut Down!")
    elif get_distribution_mode() == DISTRIBUTION_MODE.INTERCONNECTION:
        logging.warning("Shut Down!")
    else:
        raise Exception(f"Illegal distribute mode, only support ({DISTRIBUTION_MODE})")


def _resolve_args(*args, **kwargs):
    arg_flat, arg_tree = jax.tree_util.tree_flatten((args, kwargs))
    refs = {
        pos: arg for pos, arg in enumerate(arg_flat) if isinstance(arg, ray.ObjectRef)
    }

    actual_vals = ray.get(list(refs.values()))
    for pos, actual_val in zip(refs.keys(), actual_vals):
        arg_flat[pos] = actual_val

    args, kwargs = jax.tree_util.tree_unflatten(arg_tree, arg_flat)
    return args, kwargs


class RemoteFunctionWrapper(RemoteFunction):
    def _remote(self, *args, **kwargs):
        args, kwargs = _resolve_args(*args, **kwargs)
        return super()._remote(*args, **kwargs)

    def party(self, party: str):
        self.party = party
        if 'resources' in self._default_options:
            self._default_options['resources'].update({self.party: 1})
        else:
            self._default_options.update({'resources': {self.party: 1}})
        return self

    def options(self, **task_options):
        if hasattr(self, 'party') and self.party:
            if 'resources' in task_options:
                task_options['resources'].update({self.party: 1})
            else:
                task_options.update({'resources': {self.party: 1}})
        return super().options(**task_options)


class ActorClassWrapper(ActorClass):
    def party(self, party: str):
        self.party = party
        if 'resources' in self._default_options:
            self._default_options['resources'].update({self.party: 1})
        else:
            self._default_options.update({'resources': {self.party: 1}})
        return self

    def options(self, **actor_options):
        if hasattr(self, 'party') and self.party:
            if 'resources' in actor_options:
                actor_options['resources'].update({self.party: 1})
            else:
                actor_options.update({'resources': {self.party: 1}})
        return super().options(**actor_options)

    def remote(self, *args, **kwargs):
        args, kwargs = _resolve_args(*args, **kwargs)
        return super().remote(*args, **kwargs)


def _make_actor(cls, actor_options):
    if ray_version_less_than_2_0_0():
        from ray import ActorClassID
        from ray.actor import modify_class as _modify_class
    else:
        from ray.actor import ActorClassID, _modify_class

    Class = _modify_class(cls)
    _inject_tracing_into_class(Class)

    if "max_restarts" in actor_options:
        if actor_options["max_restarts"] != -1:  # -1 represents infinite restart
            # Make sure we don't pass too big of an int to C++, causing
            # an overflow.
            actor_options["max_restarts"] = min(
                actor_options["max_restarts"], ray_constants.MAX_INT64_VALUE
            )

    return ActorClassWrapper._ray_from_modified_class(
        Class,
        ActorClassID.from_random(),
        actor_options,
    )


def _make_remote(function_or_class, options):
    if inspect.isfunction(function_or_class) or _is_cython(function_or_class):
        ray_option_utils.validate_task_options(options, in_options=False)
        return RemoteFunctionWrapper(
            Language.PYTHON,
            function_or_class,
            None,
            options,
        )
    if inspect.isclass(function_or_class):
        ray_option_utils.validate_actor_options(options, in_options=False)
        return _make_actor(function_or_class, options)

    raise TypeError(
        "The @ray.remote decorator must be applied to either a function or a class."
    )


def ray_remote(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # This is the case where the decorator is just @ray.remote.
        # "args[0]" is the class or function under the decorator.
        return _make_remote(args[0], {})
    assert len(args) == 0 and len(kwargs) > 0, ray_option_utils.remote_args_error_string
    return partial(_make_remote, options=kwargs)


# Whether running in interconnection mode
def in_ic_mode() -> bool:
    return get_distribution_mode() == DISTRIBUTION_MODE.INTERCONNECTION
