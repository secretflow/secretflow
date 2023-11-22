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
from secretflow.distributed.primitive import DISTRIBUTION_MODE
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

_CROSS_SILO_COMM_BACKENDS = ['grpc', 'brpc_link']


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
    all_spu_chunks_count = []
    spu_chunks_idx = 0

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
            info, shares_chunk = x.device.outfeed_shares(x.shares_name)
            all_spu_chunks_count.append(len(shares_chunk))
            all_object_refs.append(info)
            all_object_refs.extend([s for s in shares_chunk])
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
            io_info = all_object[cur_idx]
            cur_idx += 1
            chunks_count = all_spu_chunks_count[spu_chunks_idx]
            spu_chunks_idx += 1
            shares_chunk = all_object[cur_idx : cur_idx + chunks_count]
            cur_idx += chunks_count

            new_flatten_val.append(io.reconstruct(shares_chunk, io_info))
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

    if sfd.in_ic_mode():
        return

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
    num_gpus: Optional[int] = None,
    log_to_driver=True,
    omp_num_threads: int = None,
    logging_level: str = 'info',
    cross_silo_comm_backend: str = 'grpc',
    cross_silo_comm_options: Dict = None,
    enable_waiting_for_other_parties_ready: bool = True,
    tls_config: Dict[str, Dict] = None,
    auth_manager_config: Dict = None,
    party_key_pair: Dict[str, Dict] = None,
    tee_simulation: bool = False,
    debug_mode=False,
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
                            'address': '127.0.0.1:20001',
                            # (Optional) the listen address, the `address` will
                            # be used if not prodived.
                            'listen_addr': '0.0.0.0:20001'
                        },
                        'bob': {
                            # The address for other parties.
                            'address': '127.0.0.1:20002',
                            # (Optional) the listen address, the `address` will
                            # be used if not prodived.
                            'listen_addr': '0.0.0.0:20002'
                        },
                    },
                    'self_party': 'alice'
                }

                # For bob
                {
                    'parties': {
                        'alice': {
                            # The address for other parties.
                            'address': '127.0.0.1:20001',
                            # (Optional) the listen address, the `address` will
                            # be used if not prodived.
                            'listen_addr': '0.0.0.0:20001'
                        },
                        'bob': {
                            # The address for other parties.
                            'address': '127.0.0.1:20002',
                            # (Optional) the listen address, the `address` will
                            # be used if not prodived.
                            'listen_addr': '0.0.0.0:20002'
                        },
                    },
                    'self_party': 'bob'
                }
        num_cpus: Number of CPUs the user wishes to assign to each raylet.
        log_to_driver: Whether direct output of worker processes on all nodes
            to driver.
        omp_num_threads: set environment variable `OMP_NUM_THREADS`. It works
            only when address is None.
        logging_level: optional; works only in production mode.
            the logging level, could be `debug`, `info`, `warning`, `error`,
            `critical`, not case sensititive.
        cross_silo_comm_backend: works only in production mode, a string determines
            which communication backend is used. The default value is 'grpc'.
            The other available option is 'brpc_link',  which is based on brpc.
        cross_silo_comm_options: a dict describes the cross-silo communication options.
            the common options for all cross-silo communication backends.
                exit_on_sending_failure
                    whether exit when failure on cross-silo sending. If True, a SIGTERM
                    will be signaled to self if failed to send cross-silo data. The default
                    value is True.
                messages_max_size_in_bytes
                    The maximum length in bytes of cross-silo messages. The default value
                    is 500 MB. The size must be strictly less than 2GB when grpc is used.
                timeout_in_ms
                    The timeout in miliseconds of a cross-silo RPC call. It's 60000 by default.
                serializing_allowed_list
                    A dict describes the package or class list allowed for cross-silo
                    serializing(deserializating).  It's used for avoiding pickle
                    deserializing execution attack. E.g.

                    .. code:: python

                        {
                            "numpy.core.numeric": ["*"],
                            "numpy": ["dtype"],
                        }

            when cross-silo backend is `grpc`, the following options can be configured additionally.
                grpc_retry_policy
                    a dict descibes the retry policy for cross silo rpc call.
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
                grpc_channel_options
                    A list of key-value pairs to configure the underlying gRPC Core
                    channel or server object, please refer to
                    `channel_arguments <https://grpc.github.io/grpc/python/glossary.html#term-channel_arguments>`_.

            when cross-silo backend is `brpc_link`, the following options can be configured additionally.
                1. connect_retry_times
                2. connect_retry_interval_ms
                3. recv_timeout_ms
                4. http_max_payload_size
                5. http_timeout_ms
                6. throttle_window_size
                7. brpc_channel_protocol
                    please refer to `protocols <https://github.com/apache/brpc/blob/master/docs/en/client.md#protocols>`_.
                8. brpc_channel_connection_type
                    please refer to `connection-type <https://github.com/apache/brpc/blob/master/docs/en/client.md#connection-type>`_.

        enable_waiting_for_other_parties_ready: wait for other parties ready if True.
            When cross-silo backend is `brpc_link`, you can set `connect_retry_times`
            and `connect_retry_interval_ms` in `cross_silo_comm_options` to determine
            the waiting time.

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

            .. code:: python

                # For alice
                {
                    'alice': {
                        'public_key': 'RSA public key of alice in pem.',
                        'private_key': 'RSA private key of alice in pem.',
                    }
                }

                # For bob
                {
                    'bob': {
                        'public_key': 'RSA public key of bob in pem.',
                        'private_key': 'RSA private key of bob in pem.',
                    }
                }
        tee_simulation: optional, enable TEE simulation if True.
            When simulation is enabled, the remote attestation for auth manager
            will be ignored. This is for test only and keep it False when for production.
        debug_mode: Whether to enable debug mode. In debug mode, single-process simulation
                    will be used instead of ray scheduling, and lazy mode will be changed to
                    synchronous mode to facilitate debugging.and will use PYU to simulate SPU device
                    ONLY DEBUG!

        **kwargs: see :py:meth:`ray.init` parameters.
    """
    set_logging_level(logging_level)
    simluation_mode = True if parties else False
    sfd.active_sf_cluster()
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
        num_cpus = None
        logging.warning(
            'When connecting to an existing cluster, num_cpus must not be provided. Num_cpus is neglected at this moment.'
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
        if not isinstance(parties, (str, Tuple, List)):
            raise InvalidArgumentError('parties must be str or list of str.')
        if isinstance(parties, str):
            parties = [parties]
        else:
            assert len(set(parties)) == len(parties), f'duplicated parties {parties}.'

        if debug_mode:
            # debug mode
            sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.DEBUG)
        else:
            if cluster_config:
                raise InvalidArgumentError(
                    'Simulation mode is enabled when `parties` is provided, '
                    'but you provide `cluster_config` at the same time. '
                    '`cluster_config` is for production mode only and should be `None` in simulation mode. '
                    'Or if you want to run SecretFlow in product mode, '
                    'please keep `parties` with `None`.'
                )
            # Simulation mode
            sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.SIMULATION)
            if local_mode:
                # party resources is not for scheduler cpus, but set num_cpus for convenient.
                resources = {party: num_cpus for party in parties}
            else:
                resources = None

            if not address and omp_num_threads:
                os.environ['OMP_NUM_THREADS'] = f'{omp_num_threads}'

            ray.init(
                address,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                resources=resources,
                log_to_driver=log_to_driver,
                **kwargs,
            )
        global_state.set_parties(parties=parties)

    else:
        sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.PRODUCTION)
        global _CROSS_SILO_COMM_BACKENDS
        assert cross_silo_comm_backend.lower() in _CROSS_SILO_COMM_BACKENDS, (
            'Invalid cross_silo_comm_backend, '
            f'{_CROSS_SILO_COMM_BACKENDS} are available now.'
        )
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
        all_parties: Dict = cluster_config['parties']
        if self_party not in all_parties:
            raise InvalidArgumentError(
                f'Party {self_party} not found in cluster config parties.'
            )
        for party in all_parties.values():
            assert (
                'address' in party
            ), f'There is no address for party {party} in cluster config.'
        global_state.set_parties(parties=list(all_parties.keys()))
        global_state.set_self_party(self_party)

        if tls_config:
            _parse_tls_config(tls_config, self_party)

        ray.init(
            address,
            num_cpus=num_cpus,
            log_to_driver=log_to_driver,
            **kwargs,
        )
        cross_silo_comm_options = cross_silo_comm_options or {}
        if 'exit_on_sending_failure' not in cross_silo_comm_options:
            cross_silo_comm_options['exit_on_sending_failure'] = True
        config = {
            'cross_silo_comm': cross_silo_comm_options,
            'barrier_on_initializing': enable_waiting_for_other_parties_ready,
        }
        receiver_sender_proxy_cls = None
        if cross_silo_comm_backend.lower() == 'brpc_link':
            from fed.proxy.brpc_link.link import BrpcLinkSenderReceiverProxy

            receiver_sender_proxy_cls = BrpcLinkSenderReceiverProxy
            if enable_waiting_for_other_parties_ready:
                if 'connect_retry_times' not in config['cross_silo_comm']:
                    config['cross_silo_comm']['connect_retry_times'] = 3600
                    config['cross_silo_comm']['connect_retry_interval_ms'] = 1000
        addresses = {}
        for party, addr in all_parties.items():
            if party == self_party:
                addresses[party] = addr.get('listen_addr', addr['address'])
            else:
                addresses[party] = addr['address']
        fed.init(
            addresses=addresses,
            party=self_party,
            config=config,
            logging_level=logging_level,
            tls_config=tls_config,
            receiver_sender_proxy_cls=receiver_sender_proxy_cls,
        )


def barrier():
    if sfd.get_distribution_mode() == DISTRIBUTION_MODE.PRODUCTION:
        barriers = []
        for party in global_state.parties():
            barriers.append(PYU(party)(lambda: None)())
        reveal(barriers)


def shutdown():
    """Disconnect the worker, and terminate processes started by secretflow.init().

    This will automatically run at the end when a Python process that uses Ray exits.
    It is ok to run this twice in a row. The primary use case for this function
    is to cleanup state between tests.
    """
    barrier()
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
