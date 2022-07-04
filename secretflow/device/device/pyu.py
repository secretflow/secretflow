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

import logging
from typing import Dict, List, Tuple

import jax
import ray

from .utils import check_num_returns
from .base import Device, DeviceObject, DeviceType

_LOG_FORMAT = '%(asctime)s,%(msecs)d %(levelname)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'


class PYUObject(DeviceObject):
    """PYU device object.

    Attributes:
        data (ray.ObjectRef): Reference to underlying data.
    """

    def __init__(self, device: 'PYU', data: ray.ObjectRef):
        super().__init__(device)
        self.data = data


class PYU(Device):
    """PYU is the device doing computation in single domain.

    Essentially PYU is a python worker who can execute any python code.
    """

    def __init__(self, party: str, node: str = ""):
        """PYU contructor.

        Args:
            party (str): Party name where this device is located.
            node (str, optional): Node name where thi device is located. Defaults to "".
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

    def __call__(self, fn, *, num_returns=None, **kwargs):
        """Set up ``fn`` for scheduling to this device.

        Args:
            func: Function to be schedule to this device.
            num_returns: Number of returned PYUObject.

        Returns:
            A wrapped version of ``fn``, set up for device placement.
        """

        def wrapper(*args, **kwargs):
            args_ = self._args_check(args)
            kwargs_ = self._kwargs_check(kwargs)

            _num_returns = check_num_returns(fn) if num_returns is None else num_returns
            data = self._run.options(
                resources={self.party: 1}, num_returns=_num_returns
            ).remote(fn, *args_, **kwargs_)
            if _num_returns == 1:
                return PYUObject(self, data)
            else:
                return [PYUObject(self, datum) for datum in data]

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
        logging.basicConfig(level=logging.WARNING, format=_LOG_FORMAT)

        # Automatically parse ray Object ref. Note that if it is a dictionary key, it is not parsed.
        arg_flat, arg_tree = jax.tree_util.tree_flatten((args, kwargs))
        refs = {
            pos: arg
            for pos, arg in enumerate(arg_flat)
            if isinstance(arg, ray.ObjectRef)
        }

        actual_vals = ray.get(list(refs.values()))
        for pos, actual_val in zip(refs.keys(), actual_vals):
            arg_flat[pos] = actual_val

        args, kwargs = jax.tree_util.tree_unflatten(arg_tree, arg_flat)
        return fn(*args, **kwargs)
