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

import copy
import functools
import inspect
import logging
import os
import signal
from typing import Any, Dict

from jax.tree_util import tree_flatten, tree_unflatten

from .actor import FedActorHandle
from .call_holder import FedCallHolder
from .exception import FedRemoteError, main_thread_assert
from .global_context import (
    clear_global_context,
    get_global_context,
    init_global_context,
)
from .object import FedObject
from .proxy import BrpcLinkProxy, GrpcProxy

logger = logging.getLogger(__name__)

CROSS_SILO_COMM_BACKENDS = ['grpc', 'brpc_link']


class FedRemoteFunction:
    def __init__(self, func) -> None:
        main_thread_assert()
        self._node_party = None
        self._func_body = func
        self._options = {}
        self._fed_call_holder = None
        self._func_name = func.__name__
        logger.debug(f"FedRemoteFunction create {self._func_name} id {id(self)}")

    def party(self, party: str):
        main_thread_assert()
        self._node_party = party
        self._fed_call_holder = FedCallHolder(
            self._node_party,
            self._func_name,
            self._execute_impl,
            self._options,
        )
        return self

    def options(self, **options):
        main_thread_assert()
        self._options = options
        if self._fed_call_holder:
            self._fed_call_holder.options(**options)
        return self

    def remote(self, *args, **kwargs):
        main_thread_assert()
        if not self._node_party:
            raise ValueError("You should specify a party name on the fed function.")

        return self._fed_call_holder.internal_remote(*args, **kwargs)

    def _execute_impl(self, args, kwargs):
        return self._func_body(*args, **kwargs)


class FedRemoteClass:
    def __init__(self, cls) -> None:
        main_thread_assert()
        self._party = None
        self._cls = cls
        self._options = {}
        logger.debug(f"FedRemoteClass create {self._cls.__name__} id {id(self)}")

    def party(self, party: str):
        main_thread_assert()
        self._party = party
        return self

    def options(self, **options):
        main_thread_assert()
        self._options = options
        return self

    def remote(self, *cls_args, **cls_kwargs):
        main_thread_assert()
        fed_actor_handle = FedActorHandle(
            self._cls,
            get_global_context().get_party(),
            self._party,
            self._options,
        )
        fed_actor_handle.remote(*cls_args, **cls_kwargs)
        return fed_actor_handle


original_sigint = signal.getsignal(signal.SIGINT)


def _signal_handler(signum, _):
    if signum == signal.SIGINT:
        signal.signal(signal.SIGINT, original_sigint)
        logger.warning(
            "Stop signal received (e.g. via SIGINT/Ctrl+C), "
            "try to shutdown fed. Press CTRL+C "
            "(or send SIGINT/SIGKILL/SIGTERM) to skip."
        )
        _shutdown(intended=False, on_error=True)


RAYFED_LOG_FMT = "%(asctime)s.%(msecs)03d %(levelname)s %(filename)s:%(lineno)s [%(party)s] -- %(message)s"  # noqa

RAYFED_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(logging_level, party_val=None):
    class RecordFilter(logging.Filter):
        def __init__(self, party_val) -> None:
            self._party_val = party_val
            super().__init__("FedRecordFilter")

        def filter(self, record) -> bool:
            if not hasattr(record, "party"):
                record.party = self._party_val
            return True

    logger = logging.getLogger()

    # Remove default handlers otherwise a msg will be printed twice.
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    if type(logging_level) is str:
        logging_level = logging.getLevelName(logging_level.upper())
    logger.setLevel(logging_level)

    _formatter = logging.Formatter(fmt=RAYFED_LOG_FMT, datefmt=RAYFED_DATE_FMT)
    _filter = RecordFilter(party_val)

    _customed_handler = logging.StreamHandler()
    _customed_handler.setFormatter(_formatter)
    _customed_handler.addFilter(_filter)

    logger.addHandler(_customed_handler)


def init(
    addresses: Dict = None,
    party: str = None,
    config: Dict = {},
    tls_config: Dict = None,
    logging_level: str = "info",
    job_name: str = None,
):
    main_thread_assert()
    setup_logger(logging_level, party)
    comm_backend = config.pop('cross_silo_comm_backend', 'brpc_link').lower()
    assert comm_backend in CROSS_SILO_COMM_BACKENDS, (
        'Invalid cross_silo_comm_backend, '
        f'{CROSS_SILO_COMM_BACKENDS} are available now.'
    )
    if job_name is None:
        job_name = "_unspecified_name_"
    if comm_backend == 'brpc_link':
        if config['barrier_on_initializing']:
            if 'connect_retry_times' not in config['cross_silo_comm']:
                config['cross_silo_comm']['connect_retry_times'] = 3600
                config['cross_silo_comm']['connect_retry_interval_ms'] = 1000
        proxy = BrpcLinkProxy(
            addresses, party, job_name, tls_config, config['cross_silo_comm']
        )
    elif comm_backend == 'grpc':
        proxy = GrpcProxy(
            addresses, party, job_name, tls_config, config['cross_silo_comm']
        )
    else:
        raise RuntimeError(f'Invalid cross_silo_comm_backend {comm_backend}')

    signal.signal(signal.SIGINT, _signal_handler)
    proxy.start()
    init_global_context(job_name, party, proxy, addresses)


def _shutdown(intended, on_error):
    global_context = get_global_context(assert_none=False)
    if global_context is None:
        # Do nothing since job has not been inited or is cleaned already.
        return

    wait_for_sending = True

    if on_error and not global_context.get_local_exception() is not None:
        wait_for_sending = False

    if (
        global_context.get_remote_exception() is not None
        or global_context.get_send_exception() is not None
    ):
        wait_for_sending = False

    logger.info(
        f"Shutdowning {'intendedly' if intended else 'unintendedly'}, "
        f"wait_for_sending {wait_for_sending}"
    )

    global_context = None
    clear_global_context(wait_for_sending, on_error)

    logger.info("Shutdowned")

    if not intended or on_error:
        logger.critical("Exit now due to the previous error.")
        if "SF_UT_DO_NOT_EXIT_ENV_FLAG" not in os.environ:
            os._exit(1)


def shutdown(on_error):
    main_thread_assert()
    global_context = get_global_context(assert_none=False)
    if global_context is not None and global_context.acquire_shutdown_flag():
        global_context = None
        _shutdown(True, on_error)


# This is the decorator `@fed.remote`
def remote(*args, **kwargs):
    main_thread_assert()

    def _is_cython(obj):
        """Check if an object is a Cython function or method"""

        def check_cython(x):
            return type(x).__name__ == "cython_function_or_method"

        return check_cython(obj) or (
            hasattr(obj, "__func__") and check_cython(obj.__func__)
        )

    def _make_fed_remote(function_or_class, **options):
        if inspect.isfunction(function_or_class) or _is_cython(function_or_class):
            return FedRemoteFunction(function_or_class).options(**options)

        if inspect.isclass(function_or_class):
            return FedRemoteClass(function_or_class).options(**options)

        raise TypeError(
            "The @fed.remote decorator must be applied to either a function or a class."
        )

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # This is the case where the decorator is just @fed.remote.
        return _make_fed_remote(args[0])
    assert len(args) == 0 and len(kwargs) > 0, "Remote args error."
    return functools.partial(_make_fed_remote, **kwargs)


def get(objects: Any):
    """
    Gets the real data of the given fed_object.

    If the object is located in current party, return it immediately,
    otherwise return it after receiving the real data from the located
    party.
    """

    main_thread_assert()
    addresses = get_global_context().get_addresses()
    current_party = get_global_context().get_party()
    flattened_args, tree = tree_flatten(objects)
    indexes = []

    for i, fed_object in enumerate(flattened_args):
        if not isinstance(fed_object, FedObject):
            continue

        indexes.append(i)
        if fed_object.get_party() == current_party:
            # The code path of the fed_object is in current party, so
            # need to boardcast the data of the fed_object to other parties,
            # and then return the real data of that.
            for target_party in addresses:
                if target_party == current_party:
                    continue
                else:
                    get_global_context().send(target_party, fed_object)
        else:
            # This is the code path that the fed_object is not in current party.
            # So we should insert a `recv_op` as a barrier to receive the real
            # data from the location party of the fed_object.
            get_global_context().recv(fed_object)

    try:
        for i in indexes:
            flattened_args[i] = copy.deepcopy(flattened_args[i].get_object())
    except FedRemoteError as e:
        logger.warning(
            "Encounter RemoteError happend in other parties"
            f", error message: {e._cause}"
        )
        raise
    except Exception as e:
        get_global_context().set_local_exception(e)
        raise

    return tree_unflatten(tree, flattened_args)
