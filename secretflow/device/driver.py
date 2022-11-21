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
from typing import Any, Iterable, List, Optional, Tuple, Union

import jax
import multiprocess
import ray
from spu import Visibility

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

    Agrs:
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
        vtype = Visibility.VIS_PUBLIC if spu_vis == 'public' else Visibility.VIS_SECRET
        io = SPUIO(device.conf, device.world_size)
        meta, *shares = io.make_shares(data, vtype)
        return SPUObject(device, meta, shares)

    # TODO(@xibin.wxb): support HEU conversion.
    if isinstance(device, HEU):
        raise ValueError(
            "You cannot put data to HEU directly, "
            "try put it to PYU and then move to HEU"
        )

    raise ValueError(f'Unknown device {device}')


def reveal(func_or_object):
    """Get plaintext data from device.

    NOTE: Use this function with extreme caution, as it may cause privacy leaks.
    In SecretFlow, we recommend that data should flow between different devices
    and rarely revealed to driver. Only use this function when data dependency
    control flow occurs.

    Args:
        func_or_object: May be callable or any Python objects which contains Device objects.
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
                ref = x.device.sk_keeper.decrypt_and_decode.remote(x.data)
            all_object_refs.append(ref)
        elif isinstance(x, SPUObject):
            if isinstance(x.shares[0], ray.ObjectRef):
                all_object_refs.extend(x.shares)

    cur_idx = 0
    all_object = ray.get(all_object_refs)

    new_flatten_val = []
    for x in flatten_val:
        if isinstance(x, PYUObject) or isinstance(x, HEUObject):
            new_flatten_val.append(all_object[cur_idx])
            cur_idx += 1

        elif isinstance(x, SPUObject):
            io = SPUIO(x.device.conf, x.device.world_size)
            if isinstance(x.shares[0], ray.ObjectRef):
                shares = [all_object[cur_idx + i] for i in range(x.device.world_size)]
                new_idx = cur_idx + x.device.world_size
            else:
                shares = x.shares
                new_idx = cur_idx
            new_flatten_val.append(io.reconstruct(shares))
            cur_idx = new_idx

        else:
            new_flatten_val.append(x)

    return jax.tree_util.tree_unflatten(tree, new_flatten_val)


def wait(objects: Any):
    """Wait for device objects until all are ready or error occurrency.

    Args:
        objects: struct of device objects.
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
    num_cpus: Optional[int] = None,
    log_to_driver=True,
    omp_num_threads: int = None,
    **kwargs,
):
    """Connect to an existing Ray cluster or start one and connect to it.

    Args:
        parties: parties this node represents, e.g: 'alice', ['alice', 'bob', 'carol'].
        address:  The address of the Ray cluster to connect to. If this address
            is not provided, then a raylet, a plasma store, a plasma manager,
            and some workers will be started.
        num_cpus: Number of CPUs the user wishes to assign to each raylet. 
            Min of (cpu count, 32) will be used if not provided.
        log_to_driver: Whether direct output of worker processes on all nodes to driver.
        omp_num_threads: set environment variable `OMP_NUM_THREADS`. It works only when
            address is None.
        **kwargs: see :py:meth:`ray.init` parameters.
    """
    resources = None
    if parties is not None:
        assert address is None, 'Address should be none when parties are given.'
        if num_cpus is None:
            num_cpus = min(multiprocess.cpu_count(), 32)
        assert isinstance(
            parties, (str, Tuple, List)
        ), 'parties must be str or list of str'
        if isinstance(parties, str):
            parties = [parties]
        else:
            assert len(set(parties)) == len(parties), f'duplicated parties {parties}'

        resources = {party: num_cpus for party in parties}

    if not address and omp_num_threads:
        os.environ['OMP_NUM_THREADS'] = f'{omp_num_threads}'
    ray.init(
        address,
        num_cpus=num_cpus,
        resources=resources,
        include_dashboard=False,
        log_to_driver=log_to_driver,
        **kwargs,
    )


def shutdown():
    """Disconnect the worker, and terminate processes started by secretflow.init().

    This will automatically run at the end when a Python process that uses Ray exits.
    It is ok to run this twice in a row. The primary use case for this function
    is to cleanup state between tests.
    """
    ray.shutdown()
