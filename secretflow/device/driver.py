# Copyright 2022 Ant Group Co., Ltd.
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

import os
from functools import wraps
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import fed
import jax
import multiprocess
import ray

import secretflow.distributed as sfd
from secretflow.utils.logging import set_logging_level

from .device import (
    HEU,
    PYU,
    SPU,
    SPUIO,
    Device,
    DeviceObject,
    HEUObject,
    PYUObject,
    SPUObject,
)
from .device.base import MoveConfig


def with_device(
    dev: Device,
    *,
    num_returns: int = None,
    static_argnames: Union[str, Iterable[str], None] = None,
):
    """Set up a wrapper for scheduling function to this device.

    Args:
        dev (Device): Target device.
        num_returns (int): Number of returned DeviceObject.
        static_argnames (Union[str, Iterable[str], None]): See ``jax.jit()`` docstring.

    Examples:
        >>> p1, spu = PYU(), SPU()
        >>> # dynamic decorator
        >>> x = with_device(p1)(load_data)('alice.csv')
        >>> # static decorator
        >>> @with_device(spu)
        >>> def selu(x, alpha=1.67, lmbda=1.05):
        >>>     return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
        >>> x_ = x.to(spu)
        >>> y = selu(x_)
    """

    def wrapper(fn):
        return dev(fn, num_returns=num_returns, static_argnames=static_argnames)

    return wrapper


def to(device: Device, data: Any, spu_vis: str = 'secret'):
    """Device object conversion.

    Args:
        device (Device): Target device.
        data (Any): DeviceObject or plaintext data.
        spu_vis (str): Deivce object visibility, SPU device only.
          secret: Secret sharing with protocol spdz-2k, aby3, etc.
          public: Public sharing, which means data will be replicated to each node.

    Returns:
        DeviceObject: Target device object.
    """
    assert (
        spu_vis == 'secret' or spu_vis == 'public'
    ), f'spu_vis must be public or secret'

    if isinstance(data, DeviceObject):
        return data.to(device, MoveConfig(spu_vis=spu_vis))

    if isinstance(device, PYU):
        return device(lambda x: x)(data)

    if isinstance(device, SPU):
        raise ValueError(
            "You cannot put data to SPU directly, "
            "try put it to PYU and then move to SPU"
        )

    # TODO(@xibin.wxb): support HEU conversion.
    if isinstance(device, HEU):
        raise ValueError(
            "You cannot put data to HEU directly, "
            "try put it to PYU and then move to HEU"
        )

    raise ValueError(f'Unknown device {device}')


def reveal(func_or_object, heu_encoder=None):
    """Get plaintext data from device.

    NOTE: Use this function with extreme caution, as it may cause privacy leaks.
    In SecretFlow, we recommend that data should flow between different devices
    and rarely revealed to driver. Only use this function when data dependency
    control flow occurs.

    Args:
        func_or_object: May be callable or any Python objects which contains Device objects.
        heu_encoder: Can be heu Encoder or EncoderParams.
            This is used to replace the default encoder from config
    """
    if callable(func_or_object):

        @wraps(func_or_object)
        def wrapper(*arg, **kwargs):
            return reveal(func_or_object(*arg, **kwargs))

        return wrapper
    all_object_refs = []
    flatten_val, tree = jax.tree_util.tree_flatten(func_or_object)

    for x in flatten_val:
        if isinstance(x, PYUObject):
            all_object_refs.append(x.data)
        elif isinstance(x, HEUObject):
            if x.is_plain:
                ref = x.device.get_participant(x.location).decode.remote(x.data)
            else:
                ref = x.device.sk_keeper.decrypt_and_decode.remote(x.data, heu_encoder)
            all_object_refs.append(ref)
        elif isinstance(x, SPUObject):
            assert isinstance(
                x.shares_name[0], (ray.ObjectRef, fed.FedObject)
            ), f"shares_name in spu obj should be ObjectRef or FedObject, but got {type(x.shares_name[0])} "
            all_object_refs.append(x.meta)
            all_object_refs.extend(x.device.outfeed_shares(x.shares_name))

    cur_idx = 0
    all_object = sfd.get(all_object_refs)

    new_flatten_val = []
    for x in flatten_val:
        if isinstance(x, PYUObject) or isinstance(x, HEUObject):
            new_flatten_val.append(all_object[cur_idx])
            cur_idx += 1

        elif isinstance(x, SPUObject):
            io = SPUIO(x.device.conf, x.device.world_size)
            meta = all_object[cur_idx]
            shares = [all_object[cur_idx + i + 1] for i in range(x.device.world_size)]
            new_idx = cur_idx + x.device.world_size + 1

            new_flatten_val.append(io.reconstruct(shares, meta))
            cur_idx = new_idx
        else:
            new_flatten_val.append(x)

    return jax.tree_util.tree_unflatten(tree, new_flatten_val)


def wait(objects: Any):
    """Wait for device objects until all are ready or error occurrency.
    NOTE: This function uses reveal internally, but won't reveal result to public. So this is secure to use this as synchronization semantics.
    Args:
        objects: struct of device objects.
    
    Examples:
        >>> spu = sf.SPU()
        >>> spu_value = spu(some_function)(some_value)
        >>> alice_value = spu_value.to(alice)
        >>> # synchronization
        >>> sf.wait(alice(some_save_value_function_locally)(alice_value))
    """
    # TODO(@xibin.wxb): support HEUObject
    objs = [
        x
        for x in jax.tree_util.tree_leaves(objects)
        if isinstance(x, PYUObject) or isinstance(x, SPUObject)
    ]

    reveal([o.device(lambda o: None)(o) for o in objs])


def init(
    parties: Union[str, List[str]] = None,
    address: Optional[str] = None,
    cluster_config: Dict = None,
    num_cpus: Optional[int] = None,
    log_to_driver=True,
    omp_num_threads: int = None,
    logging_level: str = 'info',
    cross_silo_grpc_retry_policy: Dict = None,
    cross_silo_send_max_retries: int = None,
    cross_silo_serializing_allowed_list: Dict = None,
    exit_on_failure_cross_silo_sending: bool = True,
    **kwargs,
):
    """Connect to an existing Ray cluster or start one and connect to it.

    Args:
        parties: parties this node represents, e.g: 'alice', ['alice', 'bob', 'carol'].
            If parties are provided, then simulation mode will be enabled,
            which means a single ray cluster will simulate as multi parties.
            If you want to run SecretFlow in production mode, plean keep it None.
        address:  The address of the Ray cluster to connect to. If this address
            is not provided, then a local ray will be started.
        cluster_config: the cluster config of multi SecretFlow parties. Must be
            provided if you run SecretFlow in cluster mode. E.g.

            .. code:: python

                # For alice
                {
                    'parties': {
                        'alice': {
                            # The address for other parties.
                            'address': '127.0.0.1:10001',
                            # (Optional) the listen address, the `address` will
                            # be used if not prodived.
                            'listen_addr': '0.0.0.0:10001'
                        },
                        'bob': {
                            # The address for other parties.
                            'address': '127.0.0.1:10002',
                            # (Optional) the listen address, the `address` will
                            # be used if not prodived.
                            'listen_addr': '0.0.0.0:10002'
                        },
                    },
                    'self_party': alice
                }

                # For bob
                {
                    'parties': {
                        'alice': {
                            # The address for other parties.
                            'address': '127.0.0.1:10001',
                            # (Optional) the listen address, the `address` will
                            # be used if not prodived.
                            'listen_addr': '0.0.0.0:10001'
                        },
                        'bob': {
                            # The address for other parties.
                            'address': '127.0.0.1:10002',
                            # (Optional) the listen address, the `address` will
                            # be used if not prodived.
                            'listen_addr': '0.0.0.0:10002'
                        },
                    },
                    'self_party': bob
                }
        num_cpus: Number of CPUs the user wishes to assign to each raylet.
        log_to_driver: Whether direct output of worker processes on all nodes
            to driver.
        omp_num_threads: set environment variable `OMP_NUM_THREADS`. It works
            only when address is None.
        logging_level: optional; works only in production mode.
            the logging level, could be `debug`, `info`, `warning`, `error`,
            `critical`, not case sensititive.
        cross_silo_grpc_retry_policy: optional, works only in production mode.
            a dict descibes the retry policy for cross silo rpc call.
            If None, the following default retry policy will be used.
            More details please refer to
            `retry-policy <https://github.com/grpc/proposal/blob/master/A6-client-retries.md#retry-policy>`_.

            .. code:: python
                {
                    "maxAttempts": 4,
                    "initialBackoff": "0.1s",
                    "maxBackoff": "1s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": [
                        "UNAVAILABLE"
                    ]
                }
        cross_silo_send_max_retries: optional, works only in production mode.
            the max retries for sending data cross silo.
        cross_silo_serializing_allowed_list: optional, works only in production mode.
            A dict describes the package or class list allowed for cross-silo
            serializing(deserializating). It's used for avoiding pickle deserializing
            execution attack when crossing silos. E.g.

            .. code:: python
                {
                    "numpy.core.numeric": ["*"],
                    "numpy": ["dtype"],
                }
        exit_on_failure_cross_silo_sending: optional, works only in production mode.
            whether exit when failure on cross-silo sending. If True, a SIGTERM
            will be signaled to self if failed to sending cross-silo data.
        **kwargs: see :py:meth:`ray.init` parameters.
    """
    resources = None
    is_standalone = True if parties else False
    local_mode = address == 'local'
    if not local_mode:
        assert (
            num_cpus is None
        ), 'When connecting to an existing cluster, num_cpus must not be provided.'
    if local_mode and num_cpus is None:
        num_cpus = multiprocess.cpu_count()
        if is_standalone:
            # Give num_cpus a min value for better simulation.
            num_cpus = min(num_cpus, 32)
    set_logging_level(logging_level)

    if is_standalone:
        # Standalone mode
        sfd.set_production(False)
        assert isinstance(
            parties, (str, Tuple, List)
        ), 'parties must be str or list of str'
        if isinstance(parties, str):
            parties = [parties]
        else:
            assert len(set(parties)) == len(parties), f'duplicated parties {parties}'

        if local_mode:
            resources = {party: num_cpus for party in parties}
        else:
            resources = None

        if not address:
            if omp_num_threads:
                os.environ['OMP_NUM_THREADS'] = f'{omp_num_threads}'

        ray.init(
            address,
            num_cpus=num_cpus,
            resources=resources,
            log_to_driver=log_to_driver,
            **kwargs,
        )
    else:
        sfd.set_production(True)
        # cluster mode
        assert (
            cluster_config
        ), 'Must provide cluster config when running with cluster mode.'
        assert 'self_party' in cluster_config, 'Miss self_party in cluster config.'
        assert 'parties' in cluster_config, 'Miss parties in cluster config.'
        self_party = cluster_config['self_party']
        all_parties = cluster_config['parties']
        assert (
            self_party in all_parties
        ), f'Party {self_party} not found in cluster config parties.'
        fed.init(
            address=address,
            cluster=all_parties,
            party=self_party,
            log_to_driver=log_to_driver,
            num_cpus=num_cpus,
            logging_level=logging_level,
            cross_silo_grpc_retry_policy=cross_silo_grpc_retry_policy,
            cross_silo_send_max_retries=cross_silo_send_max_retries,
            cross_silo_serializing_allowed_list=cross_silo_serializing_allowed_list,
            exit_on_failure_cross_silo_sending=exit_on_failure_cross_silo_sending,
            **kwargs,
        )


def shutdown():
    """Disconnect the worker, and terminate processes started by secretflow.init().

    This will automatically run at the end when a Python process that uses Ray exits.
    It is ok to run this twice in a row. The primary use case for this function
    is to cleanup state between tests.
    """
    sfd.shutdown()
