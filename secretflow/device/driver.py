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

import inspect
import multiprocessing
from functools import wraps
from typing import Any, Iterable, List, Optional, Tuple, Union

import jax
import ray
from spu import Visibility
from secretflow.device.device.spu import CustomPyTreeNode

from .device import HEU, SPU, PYU, Device, DeviceObject, HEUObject, SPUObject, PYUObject


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

    Examples
    --------
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
        return data.to(device, spu_vis=spu_vis)

    if isinstance(device, PYU):
        return device(lambda x: x)(data)

    if isinstance(device, SPU):
        spu_vis = (
            Visibility.VIS_PUBLIC if spu_vis == 'public' else Visibility.VIS_SECRET
        )

        # NOTE: Custom pytree node with attributes that can't be pickled.
        node_builder = None
        if inspect.isfunction(data) and data.__name__ == "<lambda>":
            node_builder = data
            data = node_builder()

        value_shares = SPU.infeed(device.cluster_def, str(id(data)), data, spu_vis)
        shares, tree = value_shares[:-1], value_shares[-1]
        for i, actor in enumerate(device.actors.values()):
            actor.set_var.remote(shares[i])

        # Custom pytree node
        if node_builder is not None:
            value, _ = jax.tree_util.tree_flatten(tree)
            tree = CustomPyTreeNode(value, node_builder)

        return SPUObject(device, tree)

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
        func_or_object: May be callable, PYUObject, List[PYUOject], Dict[Any, PYUObject].
    """
    if callable(func_or_object):

        @wraps(func_or_object)
        def wrapper(*arg, **kwargs):
            return reveal(func_or_object(*arg, **kwargs))

        return wrapper

    value_flat, value_tree = jax.tree_util.tree_flatten(func_or_object)
    value_ref, value_idx = [], []
    for i, value in enumerate(value_flat):
        if isinstance(value, PYUObject):
            value_ref.append(value.data)
            value_idx.append(i)
        elif isinstance(value, SPUObject):
            shares = [value.data] if isinstance(value.data, ray.ObjectRef) else []
            shares.extend(
                [
                    actor.get_var.remote(value.data)
                    for actor in value.device.actors.values()
                ]
            )
            value_ref.extend(shares)
            value_idx.append(i)
        elif isinstance(value, HEUObject):
            if value.is_plain:
                value_ref.append(value.data)
            else:
                value_ref.append(value.device.sk_keeper.decrypt.remote(value.data))
            value_idx.append(i)

    value_obj = ray.get(value_ref)
    idx = 0
    for i in value_idx:
        if isinstance(value_flat[i], (PYUObject, HEUObject)):
            value_flat[i] = value_obj[idx]
            idx += 1
        elif isinstance(value_flat[i], SPUObject):
            if isinstance(value_flat[i].data, ray.ObjectRef):
                tree = value_obj[idx]
                idx += 1
            else:
                tree = value_flat[i].data

            device = value_flat[i].device
            value_flat[i] = SPU.outfeed(
                device.conf,
                tree,
                *value_obj[idx : idx + len(device.actors)],
            )
            idx += len(device.actors)
        else:
            assert False
    assert idx == len(value_obj)

    return jax.tree_util.tree_unflatten(value_tree, value_flat)


def wait(objects: List[Union[PYUObject, SPUObject]]):
    """Wait for pyu objects until all are ready or error occurrency.

    Args:
        objects: list of device objects.
    """
    # TODO(@xibin.wxb): support HEUObject
    assert isinstance(objects, list), f'Objects should be list but got {type(objects)}'
    reveal([o.device(lambda o: None)(o) for o in objects])


def init(
    parties: Union[str, List[str]] = None,
    address: Optional[str] = None,
    num_cpus: Optional[int] = None,
    log_to_driver=False,
    **kwargs,
):
    """Connect to an existing Ray cluster or start one and connect to it.

    Args:
        parties: parties this node represents, e.g: 'alice', ['alice', 'bob', 'carol'].
        address:  The address of the Ray cluster to connect to. If this address is not provided, then this command will
        start Redis, a raylet, a plasma store, a plasma manager, and some workers.
        num_cpus: Number of CPUs the user wishes to assign to each raylet.
        log_to_driver: Whether direct output of worker processes on all nodes to driver.
        **kwargs: see :py:meth:`ray.init` parameters.
    """
    resources = None
    if parties is not None:
        assert address is None, 'Address should be none when parties are given.'
        if num_cpus is None:
            num_cpus = multiprocessing.cpu_count()
        assert isinstance(
            parties, (str, Tuple, List)
        ), f'parties must be str or list of str'
        if isinstance(parties, str):
            parties = [parties]
        else:
            assert len(set(parties)) == len(parties), f'duplicated parties {parties}'

        resources = {party: num_cpus for party in parties}

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
