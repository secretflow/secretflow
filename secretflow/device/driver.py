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
import os
import pathlib
from functools import wraps
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import fed
import jax
import multiprocess
import ray

import secretflow.distributed as sfd
from secretflow.device import global_state
from secretflow.utils.errors import InvalidArgumentError
from secretflow.utils.logging import set_logging_level
from secretflow.utils.ray_compatibility import ray_version_less_than_2_0_0

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
    TEEUObject,
)


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


def to(device: Device, data: Any):
    """Device object conversion.

    Args:
        device (Device): Target device.
        data (Any): DeviceObject or plaintext data.

    Returns:
        DeviceObject: Target device object.
    """
    if isinstance(data, DeviceObject):
        raise InvalidArgumentError(
            'You should use `DeviceObject.to(device)` directly to'
            'transfer DeviceObject to another device.'
        )

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
        elif isinstance(x, TEEUObject):
            all_object_refs.append(x.data)
            logging.debug(f'Getting teeu data from TEEU {x.device.party}.')

    cur_idx = 0
    all_object = sfd.get(all_object_refs)

    new_flatten_val = []
    for x in flatten_val:
        if isinstance(x, (PYUObject, HEUObject, TEEUObject)):
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

    NOTE: This function uses reveal internally, but won't reveal result to
    public. So this is secure to use this as synchronization semantics.

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
        if isinstance(x, (PYUObject, SPUObject, TEEUObject))
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
    cross_silo_messages_max_size_in_bytes: int = None,
    cross_silo_serializing_allowed_list: Dict = None,
    cross_silo_timeout_in_seconds: int = 3600,
    exit_on_failure_cross_silo_sending: bool = True,
    enable_waiting_for_other_parties_ready: bool = True,
    tls_config: Dict[str, Dict] = None,
    auth_manager_config: Dict = None,
    party_key_pair: Dict[str, Dict] = None,
    tee_simulation: bool = False,
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
        cross_silo_messages_max_size_in_bytes: int, works only in production mode.
            the max number of byte for one transaction.
            The size must be strictly less than 2GB, i.e. 2 * (1024 ** 3).
        cross_silo_serializing_allowed_list: optional, works only in production mode.
            A dict describes the package or class list allowed for cross-silo
            serializing(deserializating). It's used for avoiding pickle deserializing
            execution attack when crossing silos. E.g.

            .. code:: python
                {
                    "numpy.core.numeric": ["*"],
                    "numpy": ["dtype"],
                }
        cross_silo_timeout_in_seconds: The timeout in seconds of a cross-silo RPC call.
            It's 3600 by default.
        exit_on_failure_cross_silo_sending: optional, works only in production mode.
            whether exit when failure on cross-silo sending. If True, a SIGTERM
            will be signaled to self if failed to sending cross-silo data.
        enable_waiting_for_other_parties_ready: wait for other parties ready if True.
        tls_config: optional, a dict describes the tls certificate and key infomations. E.g.

            .. code:: python
                {
                    'key': 'server key in pem.'
                    'cert': 'server certificate in pem.',
                    'ca_cert': 'root ca certificate of other parties.'
                }

        auth_manager_config: optional, a dict describes the config about authority manager
            service. Authority manager helps manage the authority of TEE data.
            This parameter is for TEE users only. An example,

            .. code:: python
                {
                    'host': 'host of authority manager service.'
                    'mr_enclave': 'mr_enclave of authority manager.',
                    'ca_cert': 'optional, root ca certificate of authority manager.'
                }
        party_key_pair: optional, a dict describes the asymmetric key pair.
            This is required for party who wants to send data to TEEU.
            E.g.

            # For alice
            .. code:: python
                {
                    'alice': {
                        'public_key': 'RSA public key of alice in pem.',
                        'private_key': 'RSA private key of alice in pem.',
                    }
                }

            # For bob
            .. code:: python
                {
                    'bob': {
                        'public_key': 'RSA public key of bob in pem.',
                        'private_key': 'RSA private key of bob in pem.',
                    }
                }
        tee_simulation: optional, enable TEE simulation if True.
            When simulation is enabled, the remote attestation for auth manager
            will be ignored. This is for test only and keep it False when for production.
        **kwargs: see :py:meth:`ray.init` parameters.
    """
    set_logging_level(logging_level)
    simluation_mode = True if parties else False
    if auth_manager_config and simluation_mode:
        raise InvalidArgumentError(
            'TEE abilities is available only in production mode.'
            'Please run SecretFlow in production mode.'
        )

    if ray_version_less_than_2_0_0():
        if address:
            local_mode = False
        else:
            local_mode = True
    else:
        local_mode = address == 'local'
    if not local_mode and num_cpus is not None:
        raise InvalidArgumentError(
            'When connecting to an existing cluster, num_cpus must not be provided.'
        )
    if local_mode and num_cpus is None:
        num_cpus = multiprocess.cpu_count()
        if simluation_mode:
            # Give num_cpus a min value for better simulation.
            num_cpus = max(num_cpus, 32)

    if party_key_pair:
        _parse_party_key_pair(party_key_pair)

    if auth_manager_config:
        if not isinstance(auth_manager_config, dict):
            raise InvalidArgumentError(
                f'auth_manager_config should be a dict but got {type(auth_manager_config)}.'
            )
        if 'host' not in auth_manager_config:
            raise InvalidArgumentError('auth_manager_config does not contain host.')
        if 'mr_enclave' not in auth_manager_config:
            raise InvalidArgumentError(
                'auth_manager_config does not contain mr_enclave.'
            )

        logging.info(f'Authority manager config is {auth_manager_config}')
        global_state.set_auth_manager_host(auth_host=auth_manager_config['host'])
        global_state.set_auth_manager_mr_enclave(
            mr_enclave=auth_manager_config['mr_enclave']
        )
        auth_ca_cert_path = auth_manager_config.get('ca_cert', None)
        if auth_ca_cert_path:
            with open(auth_ca_cert_path, 'r') as f:
                auth_ca_cert = f.read()
            global_state.set_auth_manager_ca_cert(ca_cert=auth_ca_cert)

    global_state.set_tee_simulation(tee_simulation=tee_simulation)

    if simluation_mode:
        if cluster_config:
            raise InvalidArgumentError(
                'Simulation mode is enabled when `parties` is provided, '
                'but you provide `cluster_config` at the same time. '
                '`cluster_config` is for production mode only and should be `None` in simulation mode. '
                'Or if you want to run SecretFlow in product mode, '
                'please keep `parties` with `None`.'
            )
        # Simulation mode
        sfd.set_production(False)
        if not isinstance(parties, (str, Tuple, List)):
            raise InvalidArgumentError('parties must be str or list of str.')
        if isinstance(parties, str):
            parties = [parties]
        else:
            assert len(set(parties)) == len(parties), f'duplicated parties {parties}.'

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
        if not cluster_config:
            raise InvalidArgumentError(
                'Must provide `cluster_config` when running with production mode.'
                ' Or if you want to run SecretFlow in simulation mode, you should'
                ' provide `parties` and keep `cluster_config` with `None`.'
            )
        if 'self_party' not in cluster_config:
            raise InvalidArgumentError('Miss self_party in cluster config.')
        if 'parties' not in cluster_config:
            raise InvalidArgumentError('Miss parties in cluster config.')
        self_party = cluster_config['self_party']
        all_parties = cluster_config['parties']
        if self_party not in all_parties:
            raise InvalidArgumentError(
                f'Party {self_party} not found in cluster config parties.'
            )
        global_state.set_self_party(self_party)

        if tls_config:
            _parse_tls_config(tls_config, self_party)

        fed.init(
            address=address,
            cluster=all_parties,
            party=self_party,
            log_to_driver=log_to_driver,
            num_cpus=num_cpus,
            logging_level=logging_level,
            tls_config=tls_config,
            cross_silo_grpc_retry_policy=cross_silo_grpc_retry_policy,
            cross_silo_send_max_retries=cross_silo_send_max_retries,
            cross_silo_serializing_allowed_list=cross_silo_serializing_allowed_list,
            cross_silo_messages_max_size_in_bytes=cross_silo_messages_max_size_in_bytes,
            cross_silo_timeout_in_seconds=cross_silo_timeout_in_seconds,
            exit_on_failure_cross_silo_sending=exit_on_failure_cross_silo_sending,
            enable_waiting_for_other_parties_ready=enable_waiting_for_other_parties_ready,
            **kwargs,
        )


def shutdown():
    """Disconnect the worker, and terminate processes started by secretflow.init().

    This will automatically run at the end when a Python process that uses Ray exits.
    It is ok to run this twice in a row. The primary use case for this function
    is to cleanup state between tests.
    """
    sfd.shutdown()


def _parse_tls_config(
    tls_config: Dict[str, str], party: str
) -> Dict[str, global_state.PartyCert]:
    party_certs = {}
    if set(tls_config) != set(('cert', 'key', 'ca_cert')):
        raise InvalidArgumentError(
            'You should only provide cert, key and ca_cert in tls config.'
        )
    key_path = pathlib.Path(tls_config['key'])
    cert_path = pathlib.Path(tls_config['cert'])
    root_cert_path = pathlib.Path(tls_config['ca_cert'])

    if not key_path.exists():
        raise InvalidArgumentError(f'Private key file {key_path} does not exist!')
    if not cert_path.exists():
        raise InvalidArgumentError(f'Cert file {cert_path} does not exist!')
    if not root_cert_path.exists():
        raise InvalidArgumentError(f'CA cert file {root_cert_path} does not exist!')
    party_cert = global_state.PartyCert(
        party_name=party,
        key=key_path.read_text(),
        cert=cert_path.read_text(),
        root_ca_cert=root_cert_path.read_text(),
    )
    party_certs[party] = party_cert
    global_state.set_party_certs(party_certs=party_certs)


def _parse_party_key_pair(
    party_key_pair: Dict[str, Union[Dict, str]]
) -> Dict[str, global_state.PartyCert]:
    party_key_pairs = {}
    for name, info in party_key_pair.items():
        if 'private_key' not in info or 'public_key' not in info:
            raise InvalidArgumentError(
                'You should provide private_key and public_key at the same time.'
            )
        pub_key_path = pathlib.Path(info['public_key'])
        pri_key_path = pathlib.Path(info['private_key'])

        if not pub_key_path.exists:
            raise InvalidArgumentError(f'Public key file {info["key"]} does not exist!')
        if not pri_key_path.exists:
            raise InvalidArgumentError(
                f'Private key file {info["key"]} does not exist!'
            )
        party_key_pair = global_state.PartyKeyPair(
            party_name=name,
            public_key=pub_key_path.read_text(),
            private_key=pri_key_path.read_text(),
        )
        party_key_pairs[name] = party_key_pair
    global_state.set_party_key_pairs(party_key_pairs=party_key_pairs)
