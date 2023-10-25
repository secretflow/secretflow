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
from functools import wraps
from typing import Dict, Type

import jax
import ray

import secretflow.distributed as sfd
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.utils.logging import LOG_FORMAT, get_logging_level

from . import link
from .device import PYU, Device, DeviceObject, PYUObject

_WRAPPABLE_DEVICE_OBJ: Dict[Type[DeviceObject], Type[Device]] = {PYUObject: PYU}


def _actor_wrapper(device_object_type, name, num_returns):
    def wrapper(self, *args, **kwargs):
        _num_returns = kwargs.pop('_num_returns', num_returns)
        value_flat, value_tree = jax.tree_util.tree_flatten((args, kwargs))
        for i, value in enumerate(value_flat):
            if isinstance(value, DeviceObject):
                assert (
                    value.device == self.device
                ), f'unexpected device object {value.device} self {self.device}'
                value_flat[i] = value.data
        args, kwargs = jax.tree_util.tree_unflatten(value_tree, value_flat)

        logging.debug(
            (
                f'Run method {name} of actor {self.actor_class}, num_returns='
                f'{_num_returns}, args len: {len(args)}, kwargs len: {len(kwargs)}.'
            )
        )
        handle = getattr(self.data, name)
        res = handle.options(num_returns=_num_returns).remote(*args, **kwargs)
        if _num_returns == 1:
            return device_object_type(self.device, res)
        else:
            return [device_object_type(self.device, x) for x in res]

    return wrapper


def _cls_wrapper(cls):
    def ray_get_wrapper(method):
        def wrapper(*args, **kwargs):
            arg_flat, arg_tree = jax.tree_util.tree_flatten((args, kwargs))
            refs = {
                pos: arg
                for pos, arg in enumerate(arg_flat)
                if isinstance(arg, ray.ObjectRef)
            }

            if refs:
                actual_vals = ray.get(list(refs.values()))
                for pos, actual_val in zip(refs.keys(), actual_vals):
                    arg_flat[pos] = actual_val
                args, kwargs = jax.tree_util.tree_unflatten(arg_tree, arg_flat)
            return method(*args, **kwargs)

        return wrapper

    # isfunction return True on staticmethod & normal function, no classmethod
    methods = inspect.getmembers(cls, inspect.isfunction)
    # getmembers / getattr will strip methods' staticmethod decorator.
    for name, method in methods:
        if name == '__init__':
            continue

        wrapped_method = wraps(method)(ray_get_wrapper(method))
        if isinstance(inspect.getattr_static(cls, name, None), staticmethod):
            # getattr_static return methods and strip nothing.
            wrapped_method = staticmethod(wrapped_method)
        setattr(cls, name, wrapped_method)

    return cls


def proxy(
    device_object_type: Type[DeviceObject],
    max_concurrency: int = None,
    _simulation_max_concurrency: int = None,
    num_gpus: int = 0,
):
    """Define a device class which should accept DeviceObject as method parameters and return DeviceObject.

    This proxy function mainly does the following work:
    1. Add an additional parameter `device: Device` to init method `__init__`.
    2. Wrap class methods, allow passing DeviceObject as parameters, which
    must be on the same device as the class instance.
    3. According to the `return annotation` of class methods, return the
    corresponding number of DeviceObject.

    .. code:: python

        @proxy(PYUObject)
        class Model:
            def __init__(self, builder):
                self.weights = builder()

            def build_dataset(self, x, y):
                self.dataset_x = x
                self.dataset_y = y

            def get_weights(self) -> np.ndarray:
                return self.weights

            def train_step(self, step) -> Tuple[np.ndarray, int]:
                return self.weights, 100

        alice = PYU('alice')
        model = Model(builder, device=alice)
        x, y = alice(load_data)()
        model.build_dataset(x, y)
        w = model.get_weights()
        w, n = model.train_step(10)

    Args:
        device_object_type (Type[DeviceObject]): DeviceObject type, eg. PYUObject.
        max_concurrency (int): Actor threadpool size.
        _simulation_max_concurrencty (int): Actor threadpool size only for
            simulation (single controller mode). This argument takes effect only
            when max_concurrency is None.
        num_gpus: The number of GPUs to use for training. Default is 0

    Returns:
        Callable: Wrapper function.
    """
    assert (
        device_object_type in _WRAPPABLE_DEVICE_OBJ
    ), f'{device_object_type} is not allowed to be proxy'

    def make_proxy(cls):
        ActorClass = _cls_wrapper(cls)

        class ActorProxy(device_object_type):
            def __init__(self, *args, **kwargs):
                logging.basicConfig(level=get_logging_level(), format=LOG_FORMAT)

                assert 'device' in kwargs, (
                    f'missing device argument, please specify it with '
                    f'{cls.__name__}(*args, device=d, **kwargs)'
                )
                device = kwargs['device']
                expected_device_type = _WRAPPABLE_DEVICE_OBJ[device_object_type]
                assert isinstance(device, expected_device_type), (
                    f'unexpected device type, expected: '
                    f'{expected_device_type}, got {type(device)}'
                )

                if not issubclass(cls, link.Link):
                    del kwargs['device']

                max_concur = max_concurrency
                if (
                    max_concur is None
                    and _simulation_max_concurrency is not None
                    and sfd.get_distribution_mode() == DISTRIBUTION_MODE.SIMULATION
                ):
                    max_concur = _simulation_max_concurrency

                logging.info(
                    f'Create proxy actor {ActorClass} with party {device.party}.'
                )
                data = sfd.remote(ActorClass).party(device.party)
                if max_concur is not None:
                    data = data.options(max_concurrency=max_concur)
                if num_gpus > 0:
                    data = data.options(num_gpus=num_gpus)
                    kwargs["use_gpu"] = True

                data = data.remote(*args, **kwargs)
                self.actor_class = ActorClass
                super().__init__(device, data)

        methods = inspect.getmembers(cls, inspect.isfunction)
        for name, method in methods:
            if name == '__init__':
                continue
            sig = inspect.signature(method)
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

            wrapped_method = wraps(method)(
                _actor_wrapper(
                    device_object_type, name, num_returns
                )  # DeviceObject, method_name, num_returns
            )
            setattr(ActorProxy, name, wrapped_method)

        name = f"ActorProxy({cls.__name__})"
        ActorProxy.__module__ = cls.__module__
        ActorProxy.__name__ = name
        ActorProxy.__qualname__ = name
        return ActorProxy

    return make_proxy
