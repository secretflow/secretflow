# Copyright 2023 Ant Group Co., Ltd.
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
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Union

import fed
import jax
import ray

import secretflow.distributed.primitive as sfd
from secretflow.device import global_state
from secretflow.device.device._utils import check_num_returns
from secretflow.device.device.base import Device, DeviceObject, DeviceType
from secretflow.utils.logging import LOG_FORMAT, get_logging_level


@dataclass
class TEEUData:
    """Input/output data for teeu."""

    data: Any
    """
    The underlying data, can be plaintext or ciphertext (encrypted with AES256-GCM).
    """

    data_uuid: str
    """The uuid of data for authority manager.
    """

    nonce: bytes = None
    """The nonce of AES-GCM.
    """

    aad: bytes = None
    """The associated data of AES-GCM.
    """


class TEEUObject(DeviceObject):
    """

    Attributes:
        data: a reference to `TEEUData`.
    """

    def __init__(self, device: 'TEEU', data: Union[ray.ObjectRef, fed.FedObject]):
        super().__init__(device)
        self.data = data


class TEEUWorker:
    """The teeu worker which runs inside TEE as an actor."""

    def __init__(
        self,
        auth_host: str,
        auth_mr_enclave: str,
        auth_ca_cert: str = None,
        tls_cert: str = None,
        tls_key: str = None,
        simluation: bool = False,
    ) -> None:
        from sdc.auth_frame import AuthFrame, CredentialsConf

        if auth_ca_cert:
            credentials = CredentialsConf(
                root_ca=auth_ca_cert.encode('utf-8'),
                private_key=tls_key.encode('utf-8') if tls_key else None,
                cert_chain=tls_cert.encode('utf-8') if tls_cert else None,
            )
        else:
            credentials = None
        self.auth_frame = AuthFrame(
            auth_host,
            authm_mr_enclave=auth_mr_enclave,
            conf=credentials,
            sim=simluation,
        )

    def run(self, func: Callable, *args, **kwargs) -> TEEUData:
        logging.basicConfig(level=get_logging_level(), format=LOG_FORMAT)

        logging.debug(
            f'TEEU runs function: {func}, with args len: {len(args)}, kwargs len: {len(kwargs)}.'
        )

        # Auto-unboxing the ray object.
        arg_flat, arg_tree = jax.tree_util.tree_flatten((args, kwargs))
        refs = {
            pos: arg
            for pos, arg in enumerate(arg_flat)
            if isinstance(arg, ray.ObjectRef)
        }
        actual_vals = ray.get(list(refs.values()))
        for pos, actual_val in zip(refs.keys(), actual_vals):
            arg_flat[pos] = actual_val

        # Open the TEEUData.
        teeu_data = [
            (idx, value)
            for idx, value in enumerate(arg_flat)
            if isinstance(value, TEEUData)
        ]

        from secretflow.utils.cloudpickle import (
            code_position_independent_dumps as dumps,
        )

        func_bytes = dumps(func, protocol=4)
        data_keys = self.auth_frame.get_data_keys(
            func_bytes, data_uuid_list=[o[1].data_uuid for o in teeu_data]
        )

        import ray.cloudpickle.cloudpickle as pickle
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        for idx, value in teeu_data:
            aesgcm = AESGCM(data_keys[idx])
            new_value = aesgcm.decrypt(
                nonce=value.nonce, data=value.data, associated_data=value.aad
            )
            new_value = pickle.loads(new_value)
            arg_flat[idx] = new_value

        args_, kwargs_ = jax.tree_util.tree_unflatten(arg_tree, arg_flat)
        # Return plaintext by now.
        # TODO(@zhouaihui): support returning ciphertext with authority.
        return func(*args_, **kwargs_)


def _actor_wrapper(name, num_returns):
    def wrapper(self, *args, **kwargs):
        # device object type check and unwrap
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
            return TEEUObject(self.device, res)
        else:
            return [TEEUObject(self.device, x) for x in res]

    return wrapper


def _cls_wrapper(cls):
    def ray_get_wrapper(method):
        def wrapper(self, *args, **kwargs):
            logging.basicConfig(level=get_logging_level(), format=LOG_FORMAT)

            logging.debug(
                f'TEEU runs function: {method}, with args len: {len(args)}, kwargs len: {len(kwargs)}.'
            )

            arg_flat, arg_tree = jax.tree_util.tree_flatten((args, kwargs))
            refs = {
                pos: arg
                for pos, arg in enumerate(arg_flat)
                if isinstance(arg, ray.ObjectRef)
            }
            actual_vals = ray.get(list(refs.values()))
            for pos, actual_val in zip(refs.keys(), actual_vals):
                arg_flat[pos] = actual_val

            # Open the TEEUData.
            teeu_data = [
                (idx, value)
                for idx, value in enumerate(arg_flat)
                if isinstance(value, TEEUData)
            ]

            from secretflow.utils.cloudpickle import (
                code_position_independent_dumps as dumps,
            )

            class_bytes = dumps(self.__class__.__bases__[0].__bases__[0], protocol=4)
            data_keys = self.auth_frame.get_data_keys(
                class_bytes, data_uuid_list=[o[1].data_uuid for o in teeu_data]
            )

            import ray.cloudpickle.cloudpickle as pickle
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            for idx, value in teeu_data:
                aesgcm = AESGCM(data_keys[idx])
                new_value = aesgcm.decrypt(
                    nonce=value.nonce, data=value.data, associated_data=value.aad
                )
                new_value = pickle.loads(new_value)
                arg_flat[idx] = new_value

            args_, kwargs_ = jax.tree_util.tree_unflatten(arg_tree, arg_flat)

            return method(self, *args_, **kwargs_)

        return wrapper

    class ClassWithAuth(cls):
        def __init__(self, *args, **kwargs):
            logging.basicConfig(level=get_logging_level(), format=LOG_FORMAT)
            from sdc.auth_frame import AuthFrame, CredentialsConf

            auth_host = kwargs.pop('auth_host')
            auth_mr_enclave = kwargs.pop('auth_mr_enclave')
            auth_ca_cert = kwargs.pop('auth_ca_cert')
            tls_cert = kwargs.pop('tls_cert')
            tls_key = kwargs.pop('tls_key')
            simluation = kwargs.pop('simluation')
            if auth_ca_cert:
                credentials = CredentialsConf(
                    root_ca=auth_ca_cert.encode('utf-8'),
                    private_key=tls_key.encode('utf-8') if tls_key else None,
                    cert_chain=tls_cert.encode('utf-8') if tls_cert else None,
                )
            else:
                credentials = None
            self.auth_frame = AuthFrame(
                auth_host,
                authm_mr_enclave=auth_mr_enclave,
                conf=credentials,
                sim=simluation,
            )

            logging.debug(
                f'TEEU runs function: __init__, with args len: {len(args)}, kwargs len: {len(kwargs)}.'
            )

            # Auto-unboxing the ray object.
            arg_flat, arg_tree = jax.tree_util.tree_flatten((args, kwargs))
            refs = {
                pos: arg
                for pos, arg in enumerate(arg_flat)
                if isinstance(arg, ray.ObjectRef)
            }
            actual_vals = ray.get(list(refs.values()))
            for pos, actual_val in zip(refs.keys(), actual_vals):
                arg_flat[pos] = actual_val

            # Open the TEEUData.
            teeu_data = [
                (idx, value)
                for idx, value in enumerate(arg_flat)
                if isinstance(value, TEEUData)
            ]

            from secretflow.utils.cloudpickle import (
                code_position_independent_dumps as dumps,
            )

            class_bytes = dumps(self.__class__.__bases__[0].__bases__[0], protocol=4)
            data_keys = self.auth_frame.get_data_keys(
                class_bytes, data_uuid_list=[o[1].data_uuid for o in teeu_data]
            )

            import ray.cloudpickle.cloudpickle as pickle
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            for idx, value in teeu_data:
                aesgcm = AESGCM(data_keys[idx])
                new_value = aesgcm.decrypt(
                    nonce=value.nonce, data=value.data, associated_data=value.aad
                )
                new_value = pickle.loads(new_value)
                arg_flat[idx] = new_value

            args_, kwargs_ = jax.tree_util.tree_unflatten(arg_tree, arg_flat)
            super().__init__(*args_, **kwargs_)

    # isfunction return True on staticmethod & normal function, no classmethod
    methods = inspect.getmembers(ClassWithAuth, inspect.isfunction)
    # getmembers / getattr will strip methods' staticmethod decorator.
    for name, method in methods:
        if name == '__init__':
            continue

        wrapped_method = wraps(method)(ray_get_wrapper(method))
        if isinstance(inspect.getattr_static(ClassWithAuth, name, None), staticmethod):
            # getattr_static return methods and strip nothing.
            wrapped_method = staticmethod(wrapped_method)
        setattr(ClassWithAuth, name, wrapped_method)

    return ClassWithAuth


class TEEU(Device):
    """TEEU is the python processing uint of TEE.

    TEEU is designed to run python function in TEE and allows doing some
    computation safely.
    The input data of TEEU will be encrypted and nobody can open it unless
    TEEU itself. But be careful that the result of the function is plaintext
    by now, that means all parties can read the result. Please be cautious
    unless you are very aware of the risk.

    attributes:
        party: the party this TEEU belongs to.
        mr_enclave: the measurement of the TEEU enclave.

    Examples
    --------
    >>> # Here is an example showing alice and bob calculate their average.
    >>> alice = PYU('alice')
    >>> bob = PYU('bob')
    >>> teeu = TEEU('carol', mr_enclave='the mr_enclave of TEEU.')
    >>> def average(data):
    >>>     return np.average(data, axis=0)
    >>> a = self.alice(lambda: np.random.random([2, 4]))()
    >>> b = self.bob(lambda: np.random.random([2, 4]))()
    >>> a_tee = a.to(teeu, allow_funcs=average)
    >>> b_tee = b.to(teeu, allow_funcs=average)
    >>> avg_val = teeu(average)([a_tee, b_tee])
    """

    def __init__(self, party: str, mr_enclave: str):
        """Init function.

        Args:
            party: the party this TEEU belongs to.
            mr_enclave: a hex string representing the measurement of the TEEU enclave.
        """
        super().__init__(DeviceType.TEEU)
        self.party = party
        self.mr_enclave = mr_enclave
        party_cert = global_state.party_certs().get(party, global_state.PartyCert())
        self.worker = (
            sfd.remote(TEEUWorker)
            .party(self.party)
            .remote(
                auth_host=global_state.auth_manager_host(),
                auth_mr_enclave=global_state.auth_manager_mr_enclave(),
                auth_ca_cert=global_state.auth_manager_ca_cert(),
                tls_cert=party_cert.cert,
                tls_key=party_cert.key,
                simluation=global_state.tee_simulation(),
            )
        )

    def __eq__(self, other):
        return type(other) == type(self) and str(other) == str(self)

    def __call__(
        self, class_or_func, *, num_returns: int = None, **kwargs
    ) -> TEEUObject:
        if inspect.isclass(class_or_func):
            return self._class_wrapper(class_or_func, self.party, num_returns, **kwargs)
        else:
            return self._func_wrapper(class_or_func, num_returns, **kwargs)

    def _func_wrapper(self, func, num_returns: int = None, **kwargs) -> TEEUObject:
        def wrapper(*args, **kwargs):
            def try_get_data(arg, device):
                if isinstance(arg, DeviceObject):
                    assert (
                        arg.device == device
                    ), f"receive argument {arg} in different device."
                    return arg.data

            args_, kwargs_ = jax.tree_util.tree_map(
                lambda arg: try_get_data(arg, self), (args, kwargs)
            )

            _num_returns = (
                check_num_returns(func) if num_returns is None else num_returns
            )

            data = self.worker.run.options(num_returns=_num_returns).remote(
                func, *args_, **kwargs_
            )
            logging.debug(
                (
                    f'TEEU remote function: {func}, num_returns={num_returns}, '
                    f'args len: {len(args)}, kwargs len: {len(kwargs)}.'
                )
            )
            if _num_returns == 1:
                return TEEUObject(self, data)
            else:
                return [TEEUObject(self, datum) for datum in data]

        return wrapper

    def _class_wrapper(
        self, cls, party: str, num_returns: int = None, **kwargs
    ) -> TEEUObject:
        device = self
        ActorClass = _cls_wrapper(cls)

        class ActorProxy(TEEUObject):
            def __init__(self, *args, **kwargs):
                logging.basicConfig(level=get_logging_level(), format=LOG_FORMAT)

                methods = inspect.getmembers(ActorClass, inspect.isfunction)
                for name, method in methods:
                    if name == '__init__':
                        logging.info(f'Create proxy actor {inspect.signature(method)}.')
                logging.info(f'Create proxy actor {ActorClass} with party {party}.')
                data = sfd.remote(ActorClass).party(party)

                party_cert = global_state.party_certs().get(
                    party, global_state.PartyCert()
                )
                kwargs['auth_host'] = global_state.auth_manager_host()
                kwargs['auth_mr_enclave'] = global_state.auth_manager_mr_enclave()
                kwargs['auth_ca_cert'] = global_state.auth_manager_ca_cert()
                kwargs['tls_cert'] = party_cert.cert
                kwargs['tls_key'] = party_cert.key
                kwargs['simluation'] = global_state.tee_simulation()

                args, kwargs = jax.tree_util.tree_map(
                    lambda arg: arg.data if isinstance(arg, DeviceObject) else arg,
                    (args, kwargs),
                )
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
            wrapped_method = wraps(method)(_actor_wrapper(name, num_returns))
            setattr(ActorProxy, name, wrapped_method)

        name = f"ActorProxy({cls.__name__})"
        ActorProxy.__module__ = cls.__module__
        ActorProxy.__name__ = name
        ActorProxy.__qualname__ = name
        return ActorProxy
