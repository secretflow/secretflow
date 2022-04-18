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
import logging
from typing import Dict, List, Tuple

import jax
import ray

from .base import Device, DeviceObject, DeviceType

_LOG_FORMAT = '%(asctime)s,%(msecs)d %(levelname)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'


def _check_num_returns(fn):
    # inspect.signature fails on some builtin method (e.g. numpy.random.rand).
    # You can wrap a self define function which calls builtin function inside
    # with return annotation to get multi returns for now.
    if inspect.isbuiltin(fn):
        sig = inspect.signature(lambda *arg, **kwargs: fn(*arg, **kwargs))
    else:
        sig = inspect.signature(fn)

    if sig.return_annotation is None or sig.return_annotation == sig.empty:
        num_returns = 1
    else:
        if hasattr(sig.return_annotation, '_name') and sig.return_annotation._name == 'Tuple':
            num_returns = len(sig.return_annotation.__args__)
        elif isinstance(sig.return_annotation, tuple):
            num_returns = len(sig.return_annotation)
        else:
            num_returns = 1

    return num_returns


class PYUObject(DeviceObject):
    """明文Object

    明文Object即单方私有数据，只有经过加密后才可出域。

    Attributes:
        data: 明文数据
    """

    def __init__(self, device: Device, data: ray.ObjectRef):
        super().__init__(device)
        self.data = data


class PYU(Device):
    """明文计算设备

    明文计算设备即Python运行时，可以执行任意python代码。
    """

    def __init__(self, party, node=""):
        """初始化明文设备

        Args:
            party: 设备所在方
            node: 可选，设备实例
        """
        super().__init__(DeviceType.PYU)

        self.party = party
        self.node = node

    def __str__(self):
        return f'{self.party}_{self.node}'

    def __eq__(self, other):
        return type(other) == type(self) and str(other) == str(self)

    def __lt__(self, other):
        return type(other) == type(self) and str(self) < str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def __call__(self, fn, *, _num_returns=None, **kwargs):
        """Set up ``fn`` for scheduling to this device.

        Args:
            func: Function to be schedule to this device.
            _num_returns: Number of returned PYUObject.

        Returns:
            A wrapped version of ``fn``, set up for device placement.
        """

        def wrapper(*args, **kwargs):
            args_ = self._args_check(args)
            kwargs_ = self._kwargs_check(kwargs)

            num_returns = _check_num_returns(fn) if _num_returns is None else _num_returns
            data = self._run.options(resources={self.party: 1}, num_returns=num_returns).remote(fn, *args_, **kwargs_)
            if isinstance(data, list):
                return (PYUObject(self, datum) for datum in data)
            else:
                return PYUObject(self, data)

        return wrapper

    def _args_check(self, args):
        args_ = []
        for arg in args:
            if isinstance(arg, DeviceObject):
                assert arg.device == self, f"receive tensor {arg} in different device"
                args_.append(arg.data)
            elif isinstance(arg, (List, Tuple)):
                args_.append(self._args_check(arg))
            elif isinstance(arg, Dict):
                args_.append(self._kwargs_check(arg))
            else:
                args_.append(arg)

        return args_

    def _kwargs_check(self, kwargs):
        kwargs_ = {}
        for k, v in kwargs.items():
            if isinstance(v, DeviceObject):
                assert v.device == self, f"receive tensor {v} in different device"
                kwargs_[k] = v.data
            elif isinstance(v, (List, Tuple)):
                kwargs_[k] = self._args_check(v)
            elif isinstance(v, Dict):
                kwargs_[k] = self._kwargs_check(v)
            else:
                kwargs_[k] = v

        return kwargs_

    @classmethod
    @ray.remote
    def _run(fn, *args, **kwargs):
        global _LOG_FORMAT
        logging.basicConfig(level=logging.DEBUG, format=_LOG_FORMAT)

        # 自动解析ray object ref。注意，如果是字典的key，则不会解析。
        arg_flat, arg_tree = jax.tree_util.tree_flatten((args, kwargs))
        refs = {pos: arg for pos, arg in enumerate(arg_flat) if isinstance(arg, ray.ObjectRef)}
        actual_vals = ray.get(list(refs.values()))
        for pos, actual_val in zip(refs.keys(), actual_vals):
            arg_flat[pos] = actual_val
        args, kwargs = jax.tree_util.tree_unflatten(arg_tree, arg_flat)

        return fn(*args, **kwargs)
