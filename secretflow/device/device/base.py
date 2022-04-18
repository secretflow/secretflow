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

import multiprocessing
from abc import ABC, abstractmethod
from functools import wraps
from typing import List, Tuple, Optional, Union

import jax
import jax.numpy as jnp
import ray

from .register import DeviceType, dispatch

_HEAD_NODE_TAG = '__HEAD_NODE__'


class Device(ABC):
    def __init__(self, device_type: DeviceType):
        self._device_type = device_type

    @property
    def device_type(self):
        return self._device_type

    @abstractmethod
    def __call__(self, fn):
        pass


def with_device(dev: Device):
    """将函数调度至指定的设备执行

    Agrs:
        dev: 计算设备

    Examples:
    --------
    >>> p1, ppu = PYU(), PPU()
    >>> # 动态装饰
    >>> x = with_device(p1)(load_data)('alice.csv')
    >>> # 静态装饰
    >>> @with_device(ppu)
    >>> def selu(x, alpha=1.67, lmbda=1.05):
    >>>     return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
    >>> x_ = x.to(ppu)
    >>> y = selu(x_)
    """

    def wrapper(fn):
        return dev(fn)

    return wrapper


class DeviceObject(ABC):
    def __init__(self, device: Device):
        self.device = device

    @property
    def device_type(self):
        return self.device.device_type

    def to(self, device: Device, vis='secret'):
        """Device数据转换

        Args:
            device: 目标设备
            vis: 目标device对该对象的访问权限
              secret: 目标设备密文可见
              public: 目标设备明文可见

        Returns:
            DeviceObject: 目标设备对象
        """
        return dispatch('to', self, device, vis)


def to(device: Device, data, vis='secret'):
    """Device数据转换

    Args:
        device: 目标设备
        data: 待转换数据
        vis: 目标device对该对象的访问权限
          secret: 目标设备密文可见
          public: 目标设备明文可见

    Returns:
        DeviceObject：目标设备对象
    """
    from secretflow.device.device.pyu import PYU, PYUObject

    if isinstance(data, DeviceObject):
        return data.to(device, vis=vis)

    # FIXME(@xibin.wxb): PYU should not be head node if called from worker node.
    obj = PYUObject(PYU(_HEAD_NODE_TAG), ray.put(data))
    return obj.to(device, vis=vis)


def reveal(func_or_object):
    """获取指定参数或者函数返回结果的明文。

    Args:
        func_or_object: 可以是callable，PYUObject，List[PYUOject]，Dict[Any, PYUObject]。
    -------
    """
    from secretflow.device.device.pyu import PYU
    if callable(func_or_object):
        @wraps(func_or_object)
        def wrapper(*arg, **kwargs):
            return reveal(func_or_object(*arg, **kwargs))

        return wrapper

    driver = PYU(_HEAD_NODE_TAG)
    value_flat, value_tree = jax.tree_util.tree_flatten(func_or_object)
    value_obj, value_idx = [], []
    for i, value in enumerate(value_flat):
        if isinstance(value, DeviceObject):
            value_obj.append(value.to(driver).data)
            value_idx.append(i)

    value_obj = ray.get(value_obj)
    for i, value in zip(value_idx, value_obj):
        value_flat[i] = value

    return jax.tree_util.tree_unflatten(value_tree, value_flat)


def init(parties: Union[str, List[str]], address: Optional[str] = None, num_cpus: Optional[int] = None,
         log_to_driver=False, **kwargs):
    """Connect to an existing Ray cluster or start one and connect to it.

    Args:
        parties: parties this node represents, e.g: 'alice', ['alice', 'bob', 'carol'].
        address:  The address of the Ray cluster to connect to. If this address is not provided, then this command will
        start Redis, a raylet, a plasma store, a plasma manager, and some workers.
        num_cpus: Number of CPUs the user wishes to assign to each raylet.
        log_to_driver: Whether direct output of worker processes on all nodes to driver.
        **kwargs: see ray.init parameters.
    """
    assert isinstance(parties, (str, Tuple, List)), f'parties must be str or list of str'
    if isinstance(parties, str):
        parties = [parties]
    else:
        assert len(set(parties)) == len(parties), f'duplicated parties {parties}'

    num_cpus = num_cpus if num_cpus is not None else multiprocessing.cpu_count()

    resources = {}
    for party in parties:
        resources[party] = num_cpus

    # add custom resources for head node
    if address is None:
        resources[_HEAD_NODE_TAG] = num_cpus

    ray.init(address, num_cpus=num_cpus, resources=resources, include_dashboard=False, log_to_driver=log_to_driver,
             **kwargs)
