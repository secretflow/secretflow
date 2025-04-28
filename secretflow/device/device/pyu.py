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

import jax

import secretflow.distributed as sfd
from secretflow.distributed.ray_op import resolve_args
from secretflow.utils import secure_pickle as pickle
from secretflow.utils.logging import LOG_FORMAT, get_logging_level

from ._utils import check_num_returns, get_fn_code_name
from .base import Device, DeviceObject, DeviceType


class PYUObject(DeviceObject):
    """PYU device object.

    Attributes:
        data: Reference to underlying data.
    """

    device: 'PYU'

    def __init__(
        self,
        device: 'PYU',
        data,
    ):
        super().__init__(device)
        self.data = data


class PYU(Device):
    """PYU is the device doing computation in single domain.

    Essentially PYU is a python worker who can execute any python code.
    """

    def __init__(self, party: str):
        """PYU contructor.

        Args:
            party (str): Party name where this device is located.
        """
        super().__init__(DeviceType.PYU)

        self.party = party

    def __str__(self) -> str:
        return self.party

    def __repr__(self) -> str:
        return f"PYURuntime({self.party})"

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
                    ), f"receive tensor {arg} in different device, {arg.device} vs {device}"
                    return arg.data
                return arg

            args_, kwargs_ = jax.tree_util.tree_map(
                lambda arg: try_get_data(arg, self), (args, kwargs)
            )

            _num_returns = check_num_returns(fn) if num_returns is None else num_returns

            def pyu_fn(*args, **kwargs):
                return self._run(fn, *args, **kwargs)

            pyu_fn.__name__ = f"{get_fn_code_name(fn)}@PYU({self.party})"

            data = (
                sfd.remote(pyu_fn)
                .party(self.party)
                .options(num_returns=_num_returns)
                .remote(*args_, **kwargs_)
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

    def dump(self, obj: PYUObject, path: str):
        assert obj.device == self, "obj must be owned by this device."

        def pyu_dump(data, path):
            with open(path, 'wb') as f:
                pickle.dump(data, f)

        return self.__call__(pyu_dump)(obj, path)

    def load(self, path: str):
        def pyu_load(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        return self.__call__(pyu_load)(path)

    @staticmethod
    def _run(fn, *args, **kwargs):
        logging.basicConfig(level=get_logging_level(), format=LOG_FORMAT)
        logging.debug(f'PYU runs function: {fn}')

        args, kwargs = resolve_args(*args, **kwargs)
        return fn(*args, **kwargs)
