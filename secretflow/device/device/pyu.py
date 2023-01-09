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
from typing import Union

import fed
import jax
import ray

import secretflow.distributed as sfd
from secretflow.utils.logging import LOG_FORMAT, get_logging_level

from .base import Device, DeviceObject, DeviceType


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
        if (
            hasattr(sig.return_annotation, '_name')
            and sig.return_annotation._name == 'Tuple'
        ):
            num_returns = len(sig.return_annotation.__args__)
        elif isinstance(sig.return_annotation, tuple):
            num_returns = len(sig.return_annotation)
        else:
            num_returns = 1

    return num_returns


class PYUObject(DeviceObject):
    """PYU device object.

    Attributes:
        data: Reference to underlying data.
    """

    def __init__(self, device: 'PYU', data: Union[ray.ObjectRef, fed.FedObject]):
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
            node (str, optional): Node name where the device is located. Defaults to "".
        """
        super().__init__(DeviceType.PYU)

        self.party = party
        self.node = node

    def __str__(self) -> str:
        return f'{self.party}_{self.node}'

    def __repr__(self) -> str:
        return str(self)

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
            def try_get_data(arg, device):
                if isinstance(arg, DeviceObject):
                    assert (
                        arg.device == device
                    ), f"receive tensor {arg} in different device"
                    return arg.data
                return arg

            args_, kwargs_ = jax.tree_util.tree_map(
                lambda arg: try_get_data(arg, self), (args, kwargs)
            )

            _num_returns = (
                _check_num_returns(fn) if num_returns is None else num_returns
            )
            data = (
                sfd.remote(self._run)
                .party(self.party)
                .options(num_returns=_num_returns)
                .remote(fn, *args_, **kwargs_)
            )
            logging.debug(
                (
                    f'PYU remote function: {fn}, num_returns={num_returns}, '
                    f'args len: {len(args)}, kwargs len: {len(kwargs)}.'
                )
            )
            if _num_returns == 1:
                return PYUObject(self, data)
            else:
                return [PYUObject(self, datum) for datum in data]

        return wrapper

    @staticmethod
    def _run(fn, *args, **kwargs):
        logging.basicConfig(level=get_logging_level(), format=LOG_FORMAT)
        logging.debug(f'PYU runs function: {fn}')

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
