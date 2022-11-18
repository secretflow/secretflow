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

import functools
import json
import logging
import os
import shutil
import struct
import sys
import time
import threading
import uuid
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import ray
import spu
import spu.binding._lib.libs as spu_libs
import spu.binding._lib.link as spu_link
import spu.binding.util.frontend as spu_fe
from google.protobuf import json_format
from spu.binding.util.distributed import dtype_spu_to_np, shape_spu_to_np
from spu import psi
from heu import phe

from secretflow.utils.errors import InvalidArgumentError
from secretflow.utils.ndarray_bigint import BigintNdArray

from .base import Device, DeviceObject, DeviceType
from .pyu import PYUObject
from .register import dispatch
from .type_traits import spu_datatype_to_heu, spu_fxp_size

_LINK_DESC_NAMES = [
    'connect_retry_times',
    'connect_retry_interval_ms',
    'recv_timeout_ms',
    'http_max_payload_size',
    'http_timeout_ms',
    'throttle_window_size',
    'brpc_channel_protocol',
    'brpc_channel_connection_type',
]


def _fill_link_desc_attrs(link_desc: Dict, desc: spu_link.Desc):
    if link_desc:
        for name, value in link_desc.items():
            assert (
                isinstance(name, str) and name
            ), f'Link desc param name shall be a valid string but got {type(name)}.'
            if name not in _LINK_DESC_NAMES:
                raise InvalidArgumentError(
                    f'Unsupported param {name} in link desc, '
                    f'{_LINK_DESC_NAMES} are now available only.'
                )
            setattr(desc, name, value)

    if not link_desc or 'recv_timeout_ms' not in link_desc.keys():
        # set default timeout 120s
        desc.recv_timeout_ms = 120 * 1000
    if not link_desc or 'http_timeout_ms' not in link_desc.keys():
        # set default timeout 120s
        desc.http_timeout_ms = 120 * 1000


def _plaintext_to_numpy(data: Any) -> np.ndarray:
    """try to convert anything to a jax-friendly numpy array.

    Args:
        data (Any): data

    Returns:
        np.ndarray: a numpy array.
    """

    # NOTE(junfeng): jnp.asarray would transfer int64s to int32s.
    return np.asarray(jnp.asarray(data))


@dataclass
class SPUValueMeta:
    """The metadata of a SPU value, which is a Numpy array or equivalent."""

    shape: Sequence[int]
    dtype: np.dtype
    vtype: spu.Visibility


class SPUObject(DeviceObject):
    def __init__(
        self,
        device: Device,
        meta: ray.ObjectRef,
        shares: Sequence[ray.ObjectRef],
    ):
        """SPUObject refers to a Python Object which could be flattened to a
        list of SPU Values. A SPU value is a Numpy array or equivalent.
        e.g.

        1. If referred Python object is [1,2,3]
        Then meta would be referred to a single SPUValueMeta, and shares is
        a list of referrence to pieces of share of [1,2,3].

        2. If referred Python object is {'a': 1, 'b': [3, np.array(...)]}
        The meta would be referred to something like {'a': SPUValueMeta1,
        'b': [SPUValueMeta2, SPUValueMeta3]}
        Each element of shares would be referred to something like
        {'a': share1, 'b': [share2, share3]}

        3. shares is a list of ObjectRef to share slices while these share
        slices are not necessarily located at SPU device. The data transfer
        would only happen when SPU device consumes SPU objects.

        Args:
            meta: Ref to the metadata.
            refs (Sequence[ray.ObjectRef]): Refs to shares of data.
        """
        super().__init__(device)
        self.meta = meta
        self.shares = shares


class SPUIO:
    def __init__(self, runtime_config: spu.RuntimeConfig, world_size: int) -> None:
        """A wrapper of spu.Io.

        Args:
            runtime_config (RuntimeConfig): runtime_config of SPU device.
            world_size (int): world_size of SPU device.
        """
        self.runtime_config = runtime_config
        self.world_size = world_size
        self.io = spu.Io(self.world_size, self.runtime_config)

    def make_shares(self, data: Any, vtype: spu.Visibility) -> Tuple[Any, List[Any]]:
        """Convert a Python object to meta and shares of a SPUObject.

        Args:
            data (Any): Any Python object.
            vtype (Visibility): Visibility

        Returns:
            Tuple[Any, List[Any]]: meta and shares of a SPUObject
        """
        flatten_value, tree = jax.tree_util.tree_flatten(data)
        flatten_shares = []
        flatten_meta = []
        for val in flatten_value:
            val = _plaintext_to_numpy(val)
            flatten_meta.append(SPUValueMeta(val.shape, val.dtype, vtype))
            flatten_shares.append(self.io.make_shares(val, vtype))

        return jax.tree_util.tree_unflatten(tree, flatten_meta), *[  # noqa e999
            jax.tree_util.tree_unflatten(tree, list(shares))
            for shares in list(zip(*flatten_shares))
        ]

    def reconstruct(self, shares: List[Any]) -> Any:
        """Convert shares of a SPUObject to the origin Python object.

        Args:
            shares (List[Any]): Shares

        Returns:
            Any: the origin Python object.
        """
        assert len(shares) == self.world_size
        _, tree = jax.tree_util.tree_flatten(shares[0])
        flatten_shares = []
        for share in shares:
            flatten_share, _ = jax.tree_util.tree_flatten(share)
            flatten_shares.append(flatten_share)

        flatten_value = [
            self.io.reconstruct(list(shares)) for shares in list(zip(*flatten_shares))
        ]

        return jax.tree_util.tree_unflatten(tree, flatten_value)


@unique
class SPUCompilerNumReturnsPolicy(Enum):
    """Tell SPU device how to decide num of returns of called function."""

    FROM_COMPILER = 'from_compiler'
    """num of returns is from compiler result.
    """
    FROM_USER = 'from_user'
    """If users are sure that returns is a list, they could specify the length of list.
    """
    SINGLE = 'single'
    """num of returns is fixed to 1.
    """


@ray.remote
class SPURuntime:
    def __init__(self, rank: int, cluster_def: Dict, link_desc: Dict = None):
        """wrapper of spu.Runtime.

        Args:
            rank (int): rank of runtime
            cluster_def (Dict): config of spu cluster
            link_desc (Dict, optional): link config. Defaults to None.
        """
        self.rank = rank
        self.cluster_def = cluster_def

        desc = spu_link.Desc()
        for i, node in enumerate(cluster_def['nodes']):
            if i == rank and node.get('listen_address', ''):
                desc.add_party(node['id'], node['listen_address'])
            else:
                desc.add_party(node['id'], node['address'])
        _fill_link_desc_attrs(link_desc=link_desc, desc=desc)

        self.link = spu_link.create_brpc(desc, rank)
        self.conf = json_format.Parse(
            json.dumps(cluster_def['runtime_config']), spu.RuntimeConfig()
        )
        self.runtime = spu.Runtime(self.link, self.conf)

    def run(
        self,
        num_returns_policy: SPUCompilerNumReturnsPolicy,
        out_shape,
        executable: spu.spu_pb2.ExecutableProto,
        *val,
    ):
        """run executable.

        Args:
            executable (spu_pb2.ExecutableProto): the executable.

            *inputs: input vars, need to follow the exec.input_names.

        Returns:
            List: first parts are output vars following the exec.output_names. The last item is metadata.
        """

        flatten_val, _ = jax.tree_util.tree_flatten(val)

        for name, x in zip(executable.input_names, flatten_val):
            self.runtime.set_var(name, x)

        self.runtime.run(executable)

        outputs = []
        metadata = []
        for name in executable.output_names:
            var = self.runtime.get_var(name)
            outputs.append(var)
            metadata.append(
                SPUValueMeta(
                    shape_spu_to_np(var.shape),
                    dtype_spu_to_np(var.data_type),
                    var.visibility,
                )
            )
            self.runtime.del_var(name)

        for name in executable.input_names:
            self.runtime.del_var(name)

        if num_returns_policy == SPUCompilerNumReturnsPolicy.SINGLE:
            _, out_tree = jax.tree_util.tree_flatten(out_shape)
            return jax.tree_util.tree_unflatten(
                out_tree, metadata
            ), jax.tree_util.tree_unflatten(out_tree, outputs)
        elif num_returns_policy == SPUCompilerNumReturnsPolicy.FROM_COMPILER:
            return metadata + outputs
        elif num_returns_policy == SPUCompilerNumReturnsPolicy.FROM_USER:
            _, out_tree = jax.tree_util.tree_flatten(out_shape)
            single_meta, single_share = jax.tree_util.tree_unflatten(
                out_tree, metadata
            ), jax.tree_util.tree_unflatten(out_tree, outputs)
            return *(list(single_meta)), *(list(single_share))
        else:
            raise ValueError('unsupported SPUCompilerNumReturnsPolicy.')

    def a2h(self, value, exp_heu_data_type: str, schema):
        """Convert SPUObject to HEUObject.

        Args:
            tree (PyTreeLeaf): SPUObject meta info.

            exp_heu_data_type (str): HEU data type.

        Returns:
            np.ndarray: Array of `phe.Plaintext`.
        """
        expect_st = f"semi2k.AShr<{spu.spu_pb2.FieldType.Name(self.conf.field)}>"
        assert (
            value.storage_type == expect_st
        ), f"Unsupported storage type {value.storage_type}, expected {expect_st}"

        assert spu_datatype_to_heu(value.data_type) == exp_heu_data_type, (
            f"You cannot feed {value.data_type} into this HEU since it only "
            f"supports {exp_heu_data_type}, if you want to change data type of HEU, "
            f"please modify the initial configuration of HEU."
        )

        size = spu_fxp_size(self.conf.field)
        value = BigintNdArray(
            [
                int.from_bytes(
                    value.content[i * size : (i + 1) * size],
                    sys.byteorder,
                    signed=True,
                )
                for i in range(len(value.content) // size)
            ],
            value.shape.dims,
        )

        return value.to_hnp(encoder=phe.BigintEncoder(schema))

    def psi_df(
        self,
        key: Union[str, List[str]],
        data: pd.DataFrame,
        receiver: str,
        protocol='KKRT_PSI_2PC',
        precheck_input=True,
        sort=True,
        broadcast_result=True,
        bucket_size=1 << 20,
        curve_type="CURVE_25519",
    ):
        """Private set intersection with DataFrame.

        Args:
            key (str, List[str]): Column(s) used to join.
            data (pd.DataFrame): DataFrame to be joined.
            receiver (str): Which party can get joined data, others will get None.
            protocol (str): PSI protocol, See spu.psi.PsiType.
            precheck_input (bool): Whether to check input data before join.
            sort (bool): Whether sort data by key after join.
            broadcast_result (bool): Whether to broadcast joined data to all parties.
            bucket_size (int): Specified the hash bucket size used in psi. Larger values consume more memory.
            curve_type (str): curve for ecdh psi

        Returns:
            pd.DataFrame or None: joined DataFrame.
        """
        # save key dataframe to temp file for streaming psi
        data_dir = f'.data/{self.rank}-{uuid.uuid4()}'
        os.makedirs(data_dir, exist_ok=True)
        input_path, output_path = (
            f'{data_dir}/psi-input.csv',
            f'{data_dir}/psi-output.csv',
        )
        data.to_csv(input_path, index=False)

        try:
            report = self.psi_csv(
                key,
                input_path,
                output_path,
                receiver,
                protocol,
                precheck_input,
                sort,
                broadcast_result,
                bucket_size,
                curve_type,
            )

            if report['intersection_count'] == -1:
                # can not get result, return None
                return None
            else:
                # load result dataframe from temp file
                return pd.read_csv(output_path)
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def psi_csv(
        self,
        key: Union[str, List[str]],
        input_path: str,
        output_path: str,
        receiver: str,
        protocol='KKRT_PSI_2PC',
        precheck_input=True,
        sort=True,
        broadcast_result=True,
        bucket_size=1 << 20,
        curve_type="CURVE_25519",
    ):
        """Private set intersection with csv file.

        Examples:
            >>> spu = sf.SPU(utils.cluster_def)
            >>> alice = sf.PYU('alice'), sf.PYU('bob')
            >>> input_path = {alice: '/path/to/alice.csv', bob: '/path/to/bob.csv'}
            >>> output_path = {alice: '/path/to/alice_psi.csv', bob: '/path/to/bob_psi.csv'}
            >>> spu.psi_csv(['c1', 'c2'], input_path, output_path, 'alice')

        Args:
            key (str, List[str]): Column(s) used to join.
            input_path: CSV file to be joined, comma seperated and contains header.
            output_path: Joined csv file, comma seperated and contains header.
            receiver (str): Which party can get joined data.
                Others won't generate output file and `intersection_count` get `-1`.
            protocol (str): PSI protocol.
            precheck_input (bool): Whether to check input data before join.
            sort (bool): Whether sort data by key after join.
            broadcast_result (bool): Whether to broadcast joined data to all parties.
            bucket_size (int): Specified the hash bucket size used in psi.
            Larger values consume more memory.
            curve_type (str): curve for ecdh psi

        Returns:
            Dict: PSI report output by SPU.
        """
        if isinstance(key, str):
            key = [key]

        receiver_rank = -1
        for i, node in enumerate(self.cluster_def['nodes']):
            if node['party'] == receiver:
                receiver_rank = i
                break
        assert receiver_rank >= 0, f'invalid receiver {receiver}'

        config = psi.BucketPsiConfig(
            psi_type=psi.PsiType.Value(protocol),
            broadcast_result=broadcast_result,
            receiver_rank=receiver_rank,
            input_params=psi.InputParams(
                path=input_path, select_fields=key, precheck=precheck_input
            ),
            output_params=psi.OuputParams(path=output_path, need_sort=sort),
            curve_type=curve_type,
            bucket_size=bucket_size,
        )
        report = psi.bucket_psi(self.link, config)

        party = self.cluster_def['nodes'][self.rank]['party']
        return {
            'party': party,
            'original_count': report.original_count,
            'intersection_count': report.intersection_count,
        }

    def psi_join_df(
        self,
        key: Union[str, List[str]],
        data: pd.DataFrame,
        receiver: str,
        join_party: str,
        protocol='KKRT_PSI_2PC',
        precheck_input=True,
        bucket_size=1 << 20,
        curve_type="CURVE_25519",
    ):
        """Private set intersection with DataFrame.

        Examples:
            >>> spu = sf.SPU(utils.cluster_def)
            >>> spu.psi_join_df(['c1', 'c2'], [df_alice, df_bob], 'alice', 'alice')

        Args:
            key (str, List[str]): Column(s) used to join.
            data (pd.DataFrame): DataFrame to be joined.
            receiver (str): Which party can get joined data, others will get None.
            join_party (str): party joined data
            protocol (str): PSI protocol, See spu.psi.PsiType.
            precheck_input (bool): Whether to check input data before join.
            bucket_size (int): Specified the hash bucket size used in psi. Larger values consume more memory.
            curve_type (str): curve for ecdh psi

        Returns:
            pd.DataFrame or None: joined DataFrame.
        """
        # save key dataframe to temp file for streaming psi
        data_dir = f'.data/{self.rank}-{uuid.uuid4()}'
        os.makedirs(data_dir, exist_ok=True)
        input_path, output_path = (
            f'{data_dir}/psi-input.csv',
            f'{data_dir}/psi-output.csv',
        )
        data.to_csv(input_path, index=False)

        try:
            report = self.psi_join_csv(
                key,
                input_path,
                output_path,
                receiver,
                join_party,
                protocol,
                precheck_input,
                bucket_size,
                curve_type,
            )

            if report['intersection_count'] == -1:
                # can not get result, return None
                return None
            else:
                # load result dataframe from temp file
                return pd.read_csv(output_path)
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def psi_join_csv(
        self,
        key: Union[str, List[str]],
        input_path: str,
        output_path: str,
        receiver: str,
        join_party: str,
        protocol='KKRT_PSI_2PC',
        precheck_input=True,
        bucket_size=1 << 20,
        curve_type="CURVE_25519",
    ):
        """Private set intersection with csv file.

        Examples:
            >>> spu = sf.SPU(utils.cluster_def)
            >>> alice = sf.PYU('alice'), sf.PYU('bob')
            >>> input_path = {alice: '/path/to/alice.csv', bob: '/path/to/bob.csv'}
            >>> output_path = {alice: '/path/to/alice_psi.csv', bob: '/path/to/bob_psi.csv'}
            >>> spu.psi_join_csv(['c1', 'c2'], input_path, output_path, 'alice', 'alice')

        Args:
            key (str, List[str]): Column(s) used to join.
            input_path: CSV file to be joined, comma seperated and contains header.
            output_path: Joined csv file, comma seperated and contains header.
            receiver (str): Which party can get joined data. Others won't generate output file and `intersection_count` get `-1`
            join_party (str): party joined data
            protocol (str): PSI protocol.
            precheck_input (bool): Whether to check input data before join.
            bucket_size (int): Specified the hash bucket size used in psi. Larger values consume more memory.
            curve_type (str): curve for ecdh psi

        Returns:
            Dict: PSI report output by SPU.
        """
        if isinstance(key, str):
            key = [key]

        receiver_rank = -1
        for i, node in enumerate(self.cluster_def['nodes']):
            if node['party'] == receiver:
                receiver_rank = i
                break
        assert receiver_rank >= 0, f'invalid receiver {receiver}'

        # save key dataframe to temp file for streaming psi
        data_dir = f'.data/{self.rank}-{uuid.uuid4()}'
        os.makedirs(data_dir, exist_ok=True)
        input_path1, output_path1, output_path2 = (
            f'{data_dir}/psi-input.csv',
            f'{data_dir}/psi-output.csv',
            f'{data_dir}/psi-output2.csv',
        )
        origin_table = pd.read_csv(input_path)
        table_nodup = origin_table.drop_duplicates(subset=key)

        table_nodup[key].to_csv(input_path1, index=False)

        logging.warning(
            f"origin_table size:{origin_table.shape[0]},drop_duplicates size:{table_nodup.shape[0]}"
        )

        # free table_nodup dataframe
        del table_nodup

        # psi join case, need sort and broadcast set True
        sort = True
        broadcast_result = True

        config = psi.BucketPsiConfig(
            psi_type=psi.PsiType.Value(protocol),
            broadcast_result=broadcast_result,
            receiver_rank=receiver_rank,
            input_params=psi.InputParams(
                path=input_path1, select_fields=key, precheck=precheck_input
            ),
            output_params=psi.OuputParams(path=output_path1, need_sort=sort),
            curve_type=curve_type,
            bucket_size=bucket_size,
        )
        report = psi.bucket_psi(self.link, config)

        df_psi_out = pd.read_csv(output_path1)

        join_rank = -1
        for i, node in enumerate(self.cluster_def['nodes']):
            if node['party'] == join_party:
                join_rank = i
                break
        assert join_rank >= 0, f'invalid receiver {join_party}'

        self_join = False
        if join_rank == self.rank:
            self_join = True

        df_psi_join = origin_table.join(
            df_psi_out.set_index(key), on=key, how='inner', sort="False"
        )
        df_psi_join[key].to_csv(output_path1, index=False)

        in_file_stats = os.stat(output_path1)
        in_file_bytes = in_file_stats.st_size

        # TODO: better try RAII style
        in_file = open(output_path1, "rb")
        out_file = open(output_path2, "wb")

        def send_proc():
            max_read_bytes = 20480
            read_bytes = 0
            while read_bytes < in_file_bytes:
                current_read_bytes = min(max_read_bytes, in_file_bytes - read_bytes)
                current_read = in_file.read(current_read_bytes)
                assert current_read_bytes == len(
                    current_read
                ), f'invalid recv msg {current_read_bytes}!={len(current_read)}'

                packed_bytes = struct.pack(
                    f'?i{len(current_read)}s', False, len(current_read), current_read
                )

                read_bytes += current_read_bytes

                self.link.send(self.link.next_rank(), packed_bytes)
                logging.warning(f"rank:{self.rank} send {len(packed_bytes)}")

            # send last batch
            packed_bytes = struct.pack('?is', True, 1, b'\x00')
            self.link.send(self.link.next_rank(), packed_bytes)
            logging.warning(f"rank:{self.rank} send last {len(packed_bytes)}")

        def recv_proc():
            batch_count = 0
            while True:
                recv_bytes = self.link.recv(self.link.next_rank())
                batch_count += 1
                logging.warning(f"rank:{self.rank} recv {len(recv_bytes)}")

                r1, r2, r3 = struct.unpack(f'?i{len(recv_bytes)-8}s', recv_bytes)
                assert r2 == len(r3), f'invalid recv msg {r2}!={len(r3)}'
                # check if last batch
                if r1:
                    logging.warning(f"rank:{self.rank} recv last {len(recv_bytes)}")
                    break
                out_file.write(r3)

        if self.rank == 1:
            send_proc()
            recv_proc()
        else:
            recv_proc()
            send_proc()

        in_file.close()
        out_file.close()

        out_file_stats = os.stat(output_path2)
        out_file_bytes = out_file_stats.st_size

        # check psi result file size
        if out_file_bytes > 0:
            peer_psi = pd.read_csv(output_path2)
            peer_psi.columns = key

            if self_join:
                df_psi_join = origin_table.join(
                    peer_psi.set_index(key), on=key, how='inner', sort="True"
                )
            else:
                df_psi_join = peer_psi.join(
                    origin_table.set_index(key), on=key, how='inner', sort="True"
                )
        else:
            df_psi_join = pd.DataFrame(columns=key)

        join_count = df_psi_join.shape[0]
        df_psi_join.to_csv(output_path, index=False)

        # delete tmp data dir
        shutil.rmtree(data_dir, ignore_errors=True)

        party = self.cluster_def['nodes'][self.rank]['party']

        return {
            'party': party,
            'original_count': origin_table.shape[0],
            'intersection_count': report.intersection_count,
            'join_count': join_count,
        }


def _argnames_partial_except(fn, static_argnames, kwargs):
    if static_argnames is None:
        return fn, kwargs

    assert isinstance(
        static_argnames, (str, Iterable)
    ), f'type of static_argnames is {type(static_argnames)} while str or Iterable is required here.'
    if isinstance(static_argnames, str):
        static_argnames = (static_argnames,)

    static_kwargs = {k: kwargs.pop(k) for k in static_argnames if k in kwargs}
    return functools.partial(fn, **static_kwargs), kwargs


def _generate_input_uuid(name):
    return f'{name}-input-{uuid.uuid4()}'


def _generate_output_uuid(name):
    return f'{name}-output-{uuid.uuid4()}'


@ray.remote(num_returns=2)
def _spu_compile(spu_name, fn, *meta_args, **meta_kwargs):
    meta_args, meta_kwargs = jax.tree_util.tree_map(
        lambda x: ray.get(x) if isinstance(x, ray.ObjectRef) else x,
        (meta_args, meta_kwargs),
    )

    # prepare inputs and metatdata.
    input_name = []
    input_vis = []

    def _get_input_metatdata(obj: SPUObject):
        input_name.append(_generate_input_uuid(spu_name))
        input_vis.append(obj.vtype)

    jax.tree_util.tree_map(_get_input_metatdata, (meta_args, meta_kwargs))

    try:
        executable, output_tree = spu_fe.compile(
            spu_fe.Kind.JAX,
            fn,
            meta_args,
            meta_kwargs,
            input_name,
            input_vis,
            lambda output_flat: [
                _generate_output_uuid(spu_name) for _ in range(len(output_flat))
            ],
        )
    except Exception as error:
        raise ray.exceptions.WorkerCrashedError()

    return executable, output_tree


class SPU(Device):
    def __init__(self, cluster_def: Dict, link_desc: Dict = None, name: str = 'SPU'):
        """SPU device constructor.

        Args:
            cluster_def: SPU cluster definition. More details refer to
                `SPU runtime config <https://spu.readthedocs.io/en/beta/reference/runtime_config.html>`_.

                For example

                .. code:: python

                    {
                        'nodes': [
                            {
                                'party': 'alice',
                                'id': 'local:0',
                                # The address for other peers.
                                'address': '127.0.0.1:9001',
                                # The listen address of this node.
                                # Optional. Address will be used if listen_address is empty.
                                'listen_address': ''
                            },
                            {
                                'party': 'bob',
                                'id': 'local:1',
                                'address': '127.0.0.1:9002',
                                'listen_address': ''
                            },
                        ],
                        'runtime_config': {
                            'protocol': spu.spu_pb2.SEMI2K,
                            'field': spu.spu_pb2.FM128,
                            'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
                        }
                    }
            link_desc: optional. A dict specifies the link parameters.
                Available parameters are:
                    1. connect_retry_times

                    2. connect_retry_interval_ms

                    3. recv_timeout_ms

                    4. http_max_payload_size

                    5. http_timeout_ms

                    6. throttle_window_size

                    7. brpc_channel_protocol refer to `https://github.com/apache/incubator-brpc/blob/master/docs/en/client.md#protocols`

                    8. brpc_channel_connection_type refer to `https://github.com/apache/incubator-brpc/blob/master/docs/en/client.md#connection-type`
        """
        super().__init__(DeviceType.SPU)
        self.cluster_def = cluster_def
        self.link_desc = link_desc
        self.conf = json_format.Parse(
            json.dumps(cluster_def['runtime_config']), spu.RuntimeConfig()
        )
        self.world_size = len(self.cluster_def['nodes'])
        self.name = name
        self.actors = {}
        self._task_id = -1
        self.io = SPUIO(self.conf, self.world_size)
        self.init()

    def init(self):
        """Init SPU runtime in each party"""
        for rank, node in enumerate(self.cluster_def['nodes']):
            self.actors[node['party']] = SPURuntime.options(
                resources={node['party']: 1}
            ).remote(rank, self.cluster_def, self.link_desc)

    def reset(self):
        """Reset spu to clear corrupted internal state, for test only"""
        for actor in self.actors.values():
            ray.kill(actor)
        time.sleep(0.5)
        self.init()

    def _place_arguments(self, *args, **kwargs):
        def place(obj):
            if isinstance(obj, DeviceObject):
                return obj.to(self)
            else:
                # if obj is not a DeviceObject, it should be a plaintext from
                # host program, so it's safe to mark it as VIS_PUBLIC.
                meta, *refs = self.io.make_shares(obj, spu.Visibility.VIS_PUBLIC)
                return SPUObject(self, meta, refs)

        return jax.tree_util.tree_map(place, (args, kwargs))

    def __call__(
        self,
        func: Callable,
        *,
        static_argnames: Union[str, Iterable[str], None] = None,
        num_returns_policy: SPUCompilerNumReturnsPolicy = SPUCompilerNumReturnsPolicy.SINGLE,
        user_specified_num_returns: int = 1,
    ):
        def wrapper(*args, **kwargs):
            # handle static_argnames of func
            fn, kwargs = _argnames_partial_except(func, static_argnames, kwargs)

            # convert every args to SPU objects.
            args, kwargs = self._place_arguments(*args, **kwargs)

            (meta_args, meta_kwargs) = jax.tree_util.tree_map(
                lambda x: x.meta if isinstance(x, SPUObject) else x, (args, kwargs)
            )

            num_returns = user_specified_num_returns
            meta_args = list(meta_args)

            # it's ok to choose any party to compile,
            # here we choose party 0.
            executable, out_shape = _spu_compile.options(
                resources={self.cluster_def['nodes'][0]['party']: 1}
            ).remote(self.name, fn, *meta_args, **meta_kwargs)

            if num_returns_policy == SPUCompilerNumReturnsPolicy.FROM_COMPILER:
                # Since user choose to use num of returns from compiler result,
                # the compiler result must be revealed to host.
                # Performance may hurt here.
                # However, since we only expose executable here, it's still
                # safe.
                executable, out_shape = ray.get([executable, out_shape])
                num_returns = len(executable.output_names)

            if num_returns_policy == SPUCompilerNumReturnsPolicy.SINGLE:
                num_returns = 1

            # run executable and get returns.
            outputs = [None] * self.world_size
            for i, actor in enumerate(self.actors.values()):

                (actor_args, actor_kwargs) = jax.tree_util.tree_map(
                    lambda x: x.shares[i], (args, kwargs)
                )

                val, _ = jax.tree_util.tree_flatten((actor_args, actor_kwargs))

                actor_out = actor.run.options(num_returns=2 * num_returns).remote(
                    num_returns_policy, out_shape, executable, *val
                )

                outputs[i] = actor_out

            if num_returns_policy == SPUCompilerNumReturnsPolicy.SINGLE:
                return SPUObject(self, outputs[0][0], [output[1] for output in outputs])

            else:
                all_shares = [output[num_returns:] for output in outputs]
                all_meta = outputs[0][0:num_returns]

                all_atomic_spu_objects = [
                    SPUObject(self, meta, list(share))
                    for meta, share in zip(all_meta, zip(*all_shares))
                ]

                if num_returns_policy == SPUCompilerNumReturnsPolicy.FROM_USER:
                    return all_atomic_spu_objects

                _, out_tree = jax.tree_util.tree_flatten(out_shape)
                return jax.tree_util.tree_unflatten(out_tree, all_atomic_spu_objects)

        return wrapper

    def psi_df(
        self,
        key: Union[str, List[str], Dict[Device, List[str]]],
        dfs: List['PYUObject'],
        receiver: str,
        protocol='KKRT_PSI_2PC',
        precheck_input=True,
        sort=True,
        broadcast_result=True,
        bucket_size=1 << 20,
        curve_type="CURVE_25519",
    ):
        """Private set intersection with DataFrame.

        Args:
            key (str, List[str], Dict[Device, List[str]]): Column(s) used to join.
            dfs (List[PYUObject]): DataFrames to be joined, which
            should be colocated with SPU runtimes.
            receiver (str): Which party can get joined data, others will get None.
            protocol (str): PSI protocol.
            precheck_input (bool): Whether to check input data before join.
            sort (bool): Whether sort data by key after join.
            broadcast_result (bool): Whether to broadcast joined data to all parties.
            bucket_size (int): Specified the hash bucket size used in psi.
            Larger values consume more memory.
            curve_type (str): curve for ecdh psi

        Returns:
            List[PYUObject]: Joined DataFrames with order reserved.
        """
        return dispatch(
            'psi_df',
            self,
            key,
            dfs,
            receiver,
            protocol,
            precheck_input,
            sort,
            broadcast_result,
            bucket_size,
            curve_type,
        )

    def psi_csv(
        self,
        key: Union[str, List[str], Dict[Device, List[str]]],
        input_path: Union[str, Dict[Device, str]],
        output_path: Union[str, Dict[Device, str]],
        receiver: str,
        protocol='KKRT_PSI_2PC',
        precheck_input=True,
        sort=True,
        broadcast_result=True,
        bucket_size=1 << 20,
        curve_type="CURVE_25519",
    ):
        """Private set intersection with csv file.

        Args:
            key (str, List[str], Dict[Device, List[str]]): Column(s) used to join.
            input_path: CSV files to be joined, comma seperated and contains header.
            output_path: Joined csv files, comma seperated and contains header.
            receiver (str): Which party can get joined data.
            Others won't generate output file and `intersection_count` get `-1`.
            protocol (str): PSI protocol.
            precheck_input (bool): Whether check input data before joining,
            for now, it will check if key duplicate.
            sort (bool): Whether sort data by key after joining.
            broadcast_result (bool): Whether broadcast joined data to all parties.
            bucket_size (int): Specified the hash bucket size used in psi.
            Larger values consume more memory.

        Returns:
            List[Dict]: PSI reports output by SPU with order reserved.
        """

        return dispatch(
            'psi_csv',
            self,
            key,
            input_path,
            output_path,
            receiver,
            protocol,
            precheck_input,
            sort,
            broadcast_result,
            bucket_size,
            curve_type,
        )

    def psi_join_df(
        self,
        key: Union[str, List[str], Dict[Device, List[str]]],
        dfs: List['PYUObject'],
        receiver: str,
        join_party: str,
        protocol='KKRT_PSI_2PC',
        precheck_input=True,
        bucket_size=1 << 20,
        curve_type="CURVE_25519",
    ):
        """Private set intersection with csv file.

        Args:
            key (str, List[str], Dict[Device, List[str]]): Column(s) used to join.
            dfs (List[PYUObject]): DataFrames to be joined, which should be colocated with SPU runtimes.
            receiver (str): Which party can get joined data. Others won't generate output file and `intersection_count` get `-1`
            join_party (str): party can get joined data
            protocol (str): PSI protocol.
            precheck_input (bool): Whether check input data before joining, for now, it will check if key duplicate.
            bucket_size (int): Specified the hash bucket size used in psi. Larger values consume more memory.
            curve_type (str): curve for ecdh psi

        Returns:
            List[PYUObject]: Joined DataFrames with order reserved.
        """

        return dispatch(
            'psi_join_df',
            self,
            key,
            dfs,
            receiver,
            join_party,
            protocol,
            precheck_input,
            bucket_size,
            curve_type,
        )

    def psi_join_csv(
        self,
        key: Union[str, List[str], Dict[Device, List[str]]],
        input_path: Union[str, Dict[Device, str]],
        output_path: Union[str, Dict[Device, str]],
        receiver: str,
        join_party: str,
        protocol='KKRT_PSI_2PC',
        precheck_input=True,
        bucket_size=1 << 20,
        curve_type="CURVE_25519",
    ):
        """Private set intersection with csv file.

        Args:
            key (str, List[str], Dict[Device, List[str]]): Column(s) used to join.
            input_path: CSV files to be joined, comma seperated and contains header.
            output_path: Joined csv files, comma seperated and contains header.
            receiver (str): Which party can get joined data. Others won't generate output file and `intersection_count` get `-1`
            join_party (str): party can get joined data
            protocol (str): PSI protocol.
            precheck_input (bool): Whether check input data before joining, for now, it will check if key duplicate.
            bucket_size (int): Specified the hash bucket size used in psi. Larger values consume more memory.
            curve_type (str): curve for ecdh psi

        Returns:
            List[Dict]: PSI reports output by SPU with order reserved.
        """

        return dispatch(
            'psi_join_csv',
            self,
            key,
            input_path,
            output_path,
            receiver,
            join_party,
            protocol,
            precheck_input,
            bucket_size,
            curve_type,
        )
