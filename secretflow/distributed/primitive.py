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


from typing import List, Union

from .const import DISTRIBUTION_MODE, FED_OBJECT_TYPES
from .op_context import SFOpContext
from .op_strategy import SFOpStrategy

_sf_op_context = SFOpContext(DISTRIBUTION_MODE.PRODUCTION)


def add_distribution_modes(modes: dict[DISTRIBUTION_MODE, SFOpStrategy]):
    _sf_op_context.add_distribution_modes(modes)


def set_distribution_mode(mode: DISTRIBUTION_MODE):
    _sf_op_context.set_distribution_mode(mode)


def get_distribution_mode():
    return _sf_op_context.get_distribution_mode()


def get_current_cluster_idx():
    return _sf_op_context.get_current_cluster_idx()


def active_sf_cluster():
    _sf_op_context.active_sf_cluster()


def get_cluster_available_resources():
    return _sf_op_context.get_cluster_available_resources()


def init(mode: DISTRIBUTION_MODE, **kwargs):
    _sf_op_context.set_distribution_mode(mode)
    return _sf_op_context.init(**kwargs)


def remote(*args, **kwargs):
    return _sf_op_context.remote(*args, **kwargs)


def get(
    object_refs: Union[
        FED_OBJECT_TYPES,
        List[FED_OBJECT_TYPES],
        Union[object, List[object]],
    ],
):
    return _sf_op_context.get(object_refs)


def kill(actor, *, no_restart=True):
    return _sf_op_context.kill(actor, no_restart=no_restart)


def shutdown(on_error=False):
    """Shutdown the secretflow environment.

    Args:
        on_error: optional; this is useful only in production mode (using RayFed).
            This parameter indicates whether an error has occurred on your main
            thread. Rayfed is desigend to reliably send all data to peers, but will
            cease transmission if an error is detected. However, Rayfed is not equipped
            to automatically identify errors under all circumstances, particularly
            those that affect only one party independently of others. Should you
            encounter such an error, please notify Rayfed upon shutdown, and it will
            discontinue any ongoing data transmissions if
            `continue_waiting_for_data_sending_on_error` is not True.
    """
    _sf_op_context.deactivate_sf_cluster()
    _sf_op_context.shutdown(on_error=on_error)


# Whether running in interconnection mode
def in_ic_mode() -> bool:
    return get_distribution_mode() == DISTRIBUTION_MODE.INTERCONNECTION
