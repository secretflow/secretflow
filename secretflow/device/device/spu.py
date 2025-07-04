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
import itertools
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from enum import Enum, unique
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import spu
import spu.utils.frontend as spu_fe
from heu import phe
from spu import psi

import secretflow.distributed as sfd
from secretflow.distributed import FED_OBJECT_TYPES
from secretflow.distributed.ray_op import get_obj_ref
from secretflow.utils import secure_pickle as pickle
from secretflow.utils.errors import InvalidArgumentError, YACLError
from secretflow.utils.ndarray_bigint import BigintNdArray
from secretflow.utils.progress import ProgressData

from ._utils import get_fn_code_name
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

SPU_PROTOCOLS_MAP = {
    spu.ProtocolKind.SEMI2K: 'semi2k',
    spu.ProtocolKind.CHEETAH: 'cheetah',
    spu.ProtocolKind.ABY3: 'aby3',
}


def _fill_link_ssl_opts(tls_opts: Dict, link_ssl_opts: spu.link.SSLOptions):
    for name, value in tls_opts.items():
        assert (
            isinstance(name, str) and name
        ), f'tls options name shall be a valid string but got {type(name)}.'
        if hasattr(link_ssl_opts.cert, name):
            setattr(link_ssl_opts.cert, name, value)
        if hasattr(link_ssl_opts.verify, name):
            setattr(link_ssl_opts.verify, name, value)


def _fill_link_desc_attrs(link_desc: Dict, tls_opts: Dict, desc: spu.link.Desc):
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

    if tls_opts:
        server_opts = tls_opts.get('server_ssl_opts')
        client_opts = tls_opts.get('client_ssl_opts')
        _fill_link_ssl_opts(server_opts, desc.server_ssl_opts)
        _fill_link_ssl_opts(client_opts, desc.client_ssl_opts)
        desc.enable_ssl = True


def _plaintext_to_numpy(data: Any) -> np.ndarray:
    """try to convert anything to a jax-friendly numpy array.

    Args:
        data (Any): data

    Returns:
        np.ndarray: a numpy array.
    """

    # NOTE(junfeng): jnp.asarray would transfer int64s to int32s.
    return np.asarray(jnp.asarray(data))


def shape_spu_to_np(spu_shape: spu.Shape):
    return tuple(list(spu_shape.dims))


def dtype_spu_to_np(spu_dtype):
    MAP = {
        spu.DataType.DT_F32: np.float32,
        spu.DataType.DT_F64: np.float64,
        spu.DataType.DT_I1: np.bool_,
        spu.DataType.DT_I8: np.int8,
        spu.DataType.DT_U8: np.uint8,
        spu.DataType.DT_I16: np.int16,
        spu.DataType.DT_U16: np.uint16,
        spu.DataType.DT_I32: np.int32,
        spu.DataType.DT_U32: np.uint32,
        spu.DataType.DT_I64: np.int64,
        spu.DataType.DT_U64: np.uint64,
    }
    return MAP.get(spu_dtype)


@dataclass
class SPUValueMeta:
    """The metadata of an SPU value, which is a Numpy array or equivalent."""

    # duck type for jax compile
    shape: Sequence[int]
    dtype: np.dtype
    # used in _spu_compile
    vtype: spu.Visibility

    # the following meta ensures SPU object could be consumed by SPU device.
    protocol: spu.ProtocolKind
    field: spu.FieldType
    fxp_fraction_bits: int


@dataclass
class SPUIOInfo:
    """Used in SPU IO"""

    # for complex py-object that can be flattened into multiply numpy values.
    # this index indicate which chunks belong to which value.
    start_chunk_index: int
    end_chunk_index: int

    # spu.libspu.Share.meta, ValueMetaProto
    # The main difference from SPUValueMeta is that SPUValueMeta is used to compile py function
    # and the spu runtime does not perceive SPUValueMeta
    # ValueMetaProto is used for SPU IO, py runtime does not perceive the content, so keep in binary form
    meta: bytes


class SPUObject(DeviceObject):
    def __init__(
        self,
        device: Device,
        meta: FED_OBJECT_TYPES,
        shares_name: Sequence[FED_OBJECT_TYPES],
    ):
        """SPUObject refers to a Python Object which could be flattened to a
        list of SPU Values. An SPU value is a Numpy array or equivalent.
        e.g.

        1. If referred Python object is [1,2,3]
        Then meta would be referred to a single SPUValueMeta, and shares is
        a list of referrence to pieces of share of [1,2,3].

        2. If referred Python object is {'a': 1, 'b': [3, np.array(...)]}
        The meta would be referred to something like {'a': SPUValueMeta1,
        'b': [SPUValueMeta2, SPUValueMeta3]}
        Each element of shares would be referred to something like
        {'a': share1, 'b': [share2, share3]}

        3. shares is a list of ObjectRef to share chunks while these share
        chunks are not necessarily located at SPU device. The data transfer
        would only happen when SPU device consumes SPU objects.

        Args:
            meta: FED_OBJECT_TYPES: Ref to the metadata.
            shares_name: Sequence[FED_OBJECT_TYPES]: names of shares of data in each SPU node.
        """
        super().__init__(device)
        self.meta = meta
        self.shares_name = shares_name

    def __del__(self):
        if hasattr(self, "shares_name"):
            assert len(self.shares_name) == len(self.device.actors)
            for i, actor in enumerate(self.device.actors.values()):
                try:
                    actor.del_share.remote(self.shares_name[i])
                except Exception:
                    # Python doesn't make any guarantees about when __del__ is called,
                    # actor may not exist, been GCed before this function called.
                    # This may happened when Host(Driver) progress exit.
                    pass


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

    def get_shares_chunk_count(self, data: Any, vtype: spu.Visibility) -> int:
        flatten_value, _ = jax.tree_util.tree_flatten(data)
        count = 0
        for val in flatten_value:
            val = _plaintext_to_numpy(val)
            count += self.io.get_share_chunk_count(val, vtype)

        return count

    def make_shares(
        self, data: Any, vtype: spu.Visibility
    ) -> Tuple[Any, Any, List[bytes]]:
        """Convert a Python object to meta and shares of an SPUObject.

        Args:
            data (Any): Any Python object.
            vtype (Visibility): Visibility

        Returns:
            Tuple[Any, Any, *List[bytes]]: meta and share chunks of an SPUObject
            TODO: return typing in function definition is not correct,
                  */typing.Unpack support is add in py311, not support in py38
        """
        flatten_value, tree = jax.tree_util.tree_flatten(data)
        flatten_shares_chunk = [[] for _ in range(self.world_size)]
        flatten_meta = []
        flatten_io_info = []

        if len(flatten_value) == 0:
            return data, data

        for val in flatten_value:
            val = _plaintext_to_numpy(val)
            shares_chunk = self.io.make_shares(val, vtype)
            assert (
                len(shares_chunk) == self.world_size
            ), f"{len(shares_chunk)} != {self.world_size}"
            assert (
                len(set([len(s.share_chunks) for s in shares_chunk])) == 1
            ), "count of share_chunks miss match, all shares from one val should has same count."
            flatten_meta.append(
                SPUValueMeta(
                    val.shape,
                    val.dtype,
                    vtype,
                    self.runtime_config.protocol,
                    self.runtime_config.field,
                    self.runtime_config.fxp_fraction_bits,
                )
            )
            flatten_io_info.append(
                SPUIOInfo(
                    len(flatten_shares_chunk[0]),
                    len(flatten_shares_chunk[0]) + len(shares_chunk[0].share_chunks),
                    shares_chunk[0].meta,
                )
            )
            for w in range(self.world_size):
                flatten_shares_chunk[w].extend(shares_chunk[w].share_chunks)

        return (
            jax.tree_util.tree_unflatten(tree, flatten_meta),
            jax.tree_util.tree_unflatten(tree, flatten_io_info),
            *(itertools.chain.from_iterable(flatten_shares_chunk)),
        )

    def reconstruct(
        self, shares_chunk: List[bytes], io_info: Any, meta: Any = None
    ) -> Any:
        """Convert shares of an SPUObject to the origin Python object.

        Args:
            shares (List[Any]): Shares of an SPUObject
            meta (Any): Meta of an SPUObject. If not provided, sanity check would be skipped.

        Returns:
            Any: the origin Python object.
        """
        assert (
            len(shares_chunk) % self.world_size == 0
        ), f"{len(shares_chunk)} % {self.world_size}"
        flatten_info, flatten_tree = jax.tree_util.tree_flatten(io_info)
        if meta:
            flatten_metas, _ = jax.tree_util.tree_flatten(meta)
            assert len(flatten_metas) == len(
                flatten_info
            ), f"{len(flatten_metas)} != {len(flatten_info)}"
            for m in flatten_metas:
                assert m.protocol == self.runtime_config.protocol
                assert m.field == self.runtime_config.field
                assert m.fxp_fraction_bits == self.runtime_config.fxp_fraction_bits

        chunks_count_pre_party = int(len(shares_chunk) / self.world_size)
        chunks_pre_party = [
            shares_chunk[i * chunks_count_pre_party : (i + 1) * chunks_count_pre_party]
            for i in range(self.world_size)
        ]
        flatten_value = []
        for info in flatten_info:
            shares = []
            for i in range(self.world_size):
                share = spu.libspu.Share()
                share.meta = info.meta
                share.share_chunks = chunks_pre_party[i][
                    info.start_chunk_index : info.end_chunk_index
                ]
                shares.append(share)
            flatten_value.append(self.io.reconstruct(shares))

        return jax.tree_util.tree_unflatten(flatten_tree, flatten_value)


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


class SPURuntime:
    def __init__(
        self,
        rank: int,
        cluster_def: Dict,
        link_desc: Dict = None,
        log_options: spu.logging.LogOptions = spu.logging.LogOptions(),
        id: str = None,
    ):
        """wrapper of spu.Runtime.

        Args:
            rank (int): rank of runtime
            cluster_def (Dict): config of spu cluster
            link_desc (Dict, optional): link config. Defaults to None.
            log_options (spu_logging.LogOptions, optional): spu log options.
        """
        spu.logging.setup_logging(log_options)

        self.rank = rank
        self.cluster_def = cluster_def

        desc = spu.link.Desc()
        tls_opts = None
        for i, node in enumerate(cluster_def['nodes']):
            address = node['address']
            if i == rank:
                self.party = node['party']
                tls_opts = node.get('tls_opts', None)
                if node.get('listen_address', ''):
                    address = node['listen_address']
            desc.add_party(node['party'], address)
        _fill_link_desc_attrs(link_desc=link_desc, tls_opts=tls_opts, desc=desc)
        try:
            self.link = spu.link.create_brpc(desc, rank)
        except Exception as e:
            raise YACLError(f"Failed to create link: {e}")

        # self.conf = json_format.Parse(
        #     json.dumps(cluster_def['runtime_config']), spu.RuntimeConfig()
        # )
        conf = spu.RuntimeConfig()
        conf.ParseFromJsonString(json.dumps(cluster_def['runtime_config']))
        self.conf = conf
        self.runtime = spu.Runtime(self.link, self.conf)
        self.share_seq_id = 0
        self.id = id

    def __repr__(self):
        return f"SPURuntime(device_id={self.id}, party={self.party})"

    def get_new_share_name(self) -> str:
        self.share_seq_id += 1
        return f"{self.share_seq_id}"

    def infeed_share(self, io_info: Any, *shares_chunk: List[bytes]) -> Any:
        flatten_io_info, flatten_tree = jax.tree_util.tree_flatten(io_info)
        shares_name = []
        for io_info in flatten_io_info:
            share = spu.libspu.Share()
            share.meta = io_info.meta
            share.share_chunks = shares_chunk[
                io_info.start_chunk_index : io_info.end_chunk_index
            ]

            name = self.get_new_share_name()
            logging.debug(f"new share name {name}, RT {id(self.runtime)}")
            self.runtime.set_var(name, share)
            shares_name.append(name)

        return jax.tree_util.tree_unflatten(flatten_tree, shares_name)

    def outfeed_share(self, val: Any) -> Tuple[Any, List[bytes]]:
        flatten_names, flatten_tree = jax.tree_util.tree_flatten(val)
        shares_chunk = []
        flatten_io_info = []

        if len(flatten_names) == 0:
            return val

        for name in flatten_names:
            var = self.runtime.get_var(name)
            flatten_io_info.append(
                SPUIOInfo(
                    len(shares_chunk),
                    len(shares_chunk) + len(var.share_chunks),
                    var.meta,
                )
            )
            shares_chunk.extend(var.share_chunks)

        return (
            jax.tree_util.tree_unflatten(flatten_tree, flatten_io_info),
            *shares_chunk,
        )

    def outfeed_shares_chunk_count(self, val: Any) -> int:
        flatten_names, _ = jax.tree_util.tree_flatten(val)
        chunk_count = 0
        for name in flatten_names:
            chunk_count += self.runtime.get_var_chunk_count(name)

        return chunk_count

    def del_share(self, val: Any):
        flatten_names, _ = jax.tree_util.tree_flatten(val)
        for name in flatten_names:
            assert isinstance(name, str)
            logging.debug(f"del share name {name}, RT {id(self.runtime)}")
            self.runtime.del_var(name)

    def dump(self, meta: Any, val: Any, path: Union[str, Callable]):
        flatten_names, _ = jax.tree_util.tree_flatten(val)
        shares = []
        for name in flatten_names:
            shares.append(self.runtime.get_var(name))

        if isinstance(path, str):
            from pathlib import Path

            # create parent folders.
            file = Path(path)
            file.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({'meta': meta, 'shares': shares}, f)
        else:
            assert callable(path)
            with path() as w:
                pickle.dump({'meta': meta, 'shares': shares}, w)

        return None

    def load(self, path: Union[str, Callable]) -> Any:
        if isinstance(path, str):
            with open(path, 'rb') as f:
                record = pickle.load(f, filter_type=pickle.FilterType.BLACKLIST)
        else:
            assert callable(path)
            with path() as f:
                record = pickle.load(f, filter_type=pickle.FilterType.BLACKLIST)

        meta = record['meta']
        shares = record['shares']

        shares_name = []
        for share in shares:
            name = self.get_new_share_name()
            self.runtime.set_var(name, share)
            shares_name.append(name)

        _, flatten_tree = jax.tree_util.tree_flatten(meta)

        return meta, jax.tree_util.tree_unflatten(flatten_tree, shares_name)

    def run(
        self,
        num_returns_policy: SPUCompilerNumReturnsPolicy,
        out_shape,
        executable: spu.Executable,
        *val,
    ):
        """run executable.

        Args:
            executable (spu.Executable): the executable.

            *inputs: input vars, need to follow the exec.input_names.

        Returns:
            List: first parts are output vars following the exec.output_names. The last item is metadata.
        """

        logging.debug(f"SPU({id(self)}) try running {executable.name}")

        flatten_names, _ = jax.tree_util.tree_flatten(val)
        assert len(executable.input_names) == len(flatten_names)

        executable.input_names = flatten_names

        output_names = []
        for _ in range(len(executable.output_names)):
            output_names.append(self.get_new_share_name())

        executable.output_names = output_names

        logging.debug(f"SPU({id(self)}) running {executable.name} with {flatten_names}")
        self.runtime.run(executable)

        metadata = []
        for name in output_names:
            meta = self.runtime.get_var_meta(name)
            metadata.append(
                SPUValueMeta(
                    shape_spu_to_np(meta.shape),
                    dtype_spu_to_np(meta.data_type),
                    meta.visibility,
                    self.conf.protocol,
                    self.conf.field,
                    self.conf.fxp_fraction_bits,
                )
            )

        logging.debug(
            f"SPU({id(self)}) finished {executable.name} output {output_names}"
        )

        if num_returns_policy == SPUCompilerNumReturnsPolicy.SINGLE:
            _, out_tree = jax.tree_util.tree_flatten(out_shape)
            return jax.tree_util.tree_unflatten(
                out_tree, metadata
            ), jax.tree_util.tree_unflatten(out_tree, output_names)
        elif num_returns_policy == SPUCompilerNumReturnsPolicy.FROM_COMPILER:
            return metadata + output_names
        elif num_returns_policy == SPUCompilerNumReturnsPolicy.FROM_USER:
            _, out_tree = jax.tree_util.tree_flatten(out_shape)
            single_meta, single_share = jax.tree_util.tree_unflatten(
                out_tree, metadata
            ), jax.tree_util.tree_unflatten(out_tree, output_names)

            if hasattr(single_meta, '__iter__'):
                return (*(single_meta), *(single_share))
            else:
                return single_meta, single_share
        else:
            raise ValueError('unsupported SPUCompilerNumReturnsPolicy.')

    def a2h(self, io_info, exp_heu_data_type: str, schema, *chunks):
        """Convert SPUObject to HEUObject.

        Args:
            tree (PyTreeLeaf): SPUObject meta info.

            exp_heu_data_type (str): HEU data type.

        Returns:
            np.ndarray: Array of `phe.Plaintext`.
        """
        assert isinstance(io_info, SPUIOInfo), "not support pytree for now"
        spu_meta = spu.ValueMeta()
        spu_meta.ParseFromString(io_info.meta)
        assert io_info.start_chunk_index == 0, "not support pytree for now"
        assert io_info.end_chunk_index == len(chunks), "not support pytree for now"

        expect_st = f"semi2k.AShr<{self.conf.field.name}>"
        assert (
            spu_meta.storage_type == expect_st
        ), f"Unsupported storage type {spu_meta.storage_type}, expected {expect_st}"

        assert spu_datatype_to_heu(spu_meta.data_type) == exp_heu_data_type, (
            f"You cannot feed {spu_meta.data_type} into this HEU since it only "
            f"supports {exp_heu_data_type}, if you want to change data type of HEU, "
            f"please modify the initial configuration of HEU."
        )

        size = spu_fxp_size(self.conf.field)

        def _bytes_to_pb(chunk_idx: int) -> spu.ValueChunk:
            ret = spu.ValueChunk()
            ret.ParseFromString(chunks[chunk_idx])
            return ret

        def _get_int_bytes() -> bytes:
            if len(chunks) == 0:
                return
            chunk_idx = 0
            chunk_pb = _bytes_to_pb(chunk_idx)
            chunk_idx += 1
            assert (
                chunk_pb.total_bytes % size == 0
            ), f"share size {chunk_pb.total_bytes} need align to {size}"
            total_pos = 0
            chunk_pos = 0
            int_bytes = b""
            while total_pos < chunk_pb.total_bytes:
                except_len = size - len(int_bytes)
                read_len = min(except_len, len(chunk_pb.content) - chunk_pos)
                int_bytes += chunk_pb.content[chunk_pos : chunk_pos + read_len]
                total_pos += read_len
                chunk_pos += read_len
                if len(int_bytes) == size:
                    yield int_bytes
                    int_bytes = b""
                if chunk_pos == len(chunk_pb.content) and chunk_idx < len(chunks):
                    chunk_pb = _bytes_to_pb(chunk_idx)
                    chunk_idx += 1
                    chunk_pos = 0

        value = BigintNdArray(
            [
                int.from_bytes(int_b, sys.byteorder, signed=True)
                for int_b in _get_int_bytes()
            ],
            spu_meta.shape.dims,
        )

        return value.to_hnp(encoder=phe.BigintEncoder(schema))

    def psi_df(
        self,
        key: Union[str, List[str]],
        data: pd.DataFrame,
        receiver: str,
        protocol='PROTOCOL_RR22',
        precheck_input=True,
        sort=True,
        broadcast_result=True,
        bucket_size=1 << 20,
        curve_type="CURVE_25519",
        dppsi_bob_sub_sampling=0.9,
        dppsi_epsilon=3,
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
            dppsi_bob_sub_sampling (double): bob subsampling bernoulli_distribution
                probability of dp psi
            dppsi_epsilon (double): epsilon of dp psi

        Returns:
            pd.DataFrame or None: joined DataFrame.
        """
        if isinstance(key, str):
            key = [key]

        # save key dataframe to temp file for streaming psi
        with tempfile.TemporaryDirectory() as data_dir:
            input_path, output_path = (
                f'{data_dir}/psi-input.csv',
                f'{data_dir}/psi-output.csv',
            )
            data.to_csv(input_path, index=False)

            report = self.psi(
                keys=key,
                input_path=input_path,
                output_path=output_path,
                receiver=receiver,
                protocol=protocol,
                table_keys_duplicated=not precheck_input,
                disable_alignment=not sort,
                broadcast_result=broadcast_result,
                bucket_size=bucket_size,
                ecdh_curve=curve_type,
                dppsi_bob_sub_sampling=dppsi_bob_sub_sampling,
                dppsi_epsilon=dppsi_epsilon,
            )

            if report['intersection_count'] == -1:
                # can not get result, return None
                return None
            else:
                # load result dataframe from temp file
                return pd.read_csv(output_path)

    def ub_psi(
        self,
        mode: str,
        role: str,
        input_path: str,
        keys: List[str],
        server_secret_key_path: str,
        cache_path: str,
        server_get_result: bool,
        client_get_result: bool,
        disable_alignment: bool,
        output_path: str,
        join_type: str,
        left_side: str,
        null_rep: str,
    ):
        """Unbalanced PSI.
        Args:
            mode (str): Mode of psi. One of [
                MODE_UNSPECIFIED,
                MODE_OFFLINE_GEN_CACHE,
                MODE_OFFLINE_TRANSFER_CACHE,
                MODE_OFFLINE,
                MODE_ONLINE,
                MODE_FULL
            ]
            role (str): Role of psi. one of [
                ROLE_SERVER,
                ROLE_CLIENT,
            ]
            input_path (str): Input path of psi.
            keys (List[str]): Keys of psi.
            server_secret_key_path (str): Server secret key path of psi.
            cache_path (str): Cache path of psi.
            server_get_result (bool): Server get result of psi.
            client_get_result (bool): Client get result of psi.
            disable_alignment (bool): Disable alignment of psi.
            output_path (str): Output path of psi.
        Returns:
            Dict: PSI report output by SPU.
        """

        left_side_rank = -1
        server_rank = -1
        for i, node in enumerate(self.cluster_def['nodes']):
            if node['party'] == left_side:
                left_side_rank = i
            if node['party'] == self.party and role == "ROLE_SERVER":
                server_rank = i
            else:
                server_rank = len(self.cluster_def['nodes']) - i
        if left_side_rank < 0 and left_side == "":
            # default is server
            left_side_rank = server_rank
        assert left_side_rank >= 0, f'invalid `left_side` {left_side}'

        config = spu.psi.UbPsiExecuteConfig(
            mode=spu.psi.parse_ub_psi_mode(mode),
            role=spu.psi.parse_ub_psi_role(role),
            server_receive_result=server_get_result,
            client_receive_result=client_get_result,
            cache_path=cache_path,
            input_params=spu.psi.InputParams(
                type=spu.psi.SourceType.SOURCE_TYPE_FILE_CSV,
                path=input_path,
                selected_keys=keys,
            ),
            output_params=spu.psi.OutputParams(
                type=spu.psi.SourceType.SOURCE_TYPE_FILE_CSV,
                path=output_path,
                disable_alignment=disable_alignment,
                csv_null_rep=null_rep,
            ),
            server_params=spu.psi.UbPsiServerParams(
                secret_key_path=server_secret_key_path
            ),
            join_conf=spu.psi.ResultJoinConfig(
                type=spu.psi.parse_join_type(join_type),
                left_side_rank=left_side_rank,
            ),
        )
        report = spu.psi.ub_psi_execute(config, self.link)
        return {
            'party': self.party,
            'original_count': report.original_count,
            'intersection_count': report.intersection_count,
        }

    def psi(
        self,
        keys: List[str],
        input_path: str,
        output_path: str,
        receiver: str,
        table_keys_duplicated: bool,
        output_csv_na_rep: str = "NULL",
        broadcast_result: bool = True,
        protocol: str = 'PROTOCOL_RR22',
        ecdh_curve: str = 'CURVE_FOURQ',
        join_type: str = "JOIN_TYPE_UNSPECIFIED",
        left_side: str = "",
        disable_alignment: bool = False,
        bucket_size=1 << 20,
        dppsi_bob_sub_sampling=0.9,
        dppsi_epsilon=3,
    ):
        if protocol == "PROTOCOL_DP":
            assert (
                0 < dppsi_bob_sub_sampling < 1
            ), f'invalid bob_sub_sampling({dppsi_bob_sub_sampling}) for dp-psi'
            assert 0 < dppsi_epsilon, f'invalid epsilon({dppsi_epsilon}) for dp-psi'

            config.dppsi_params = psi.DpPsiParams(
                bob_sub_sampling=dppsi_bob_sub_sampling, epsilon=dppsi_epsilon
            )

        receiver_rank = -1
        left_side_rank = -1
        for i, node in enumerate(self.cluster_def['nodes']):
            if node['party'] == receiver:
                receiver_rank = i
            if node['party'] == left_side:
                left_side_rank = i
        assert receiver_rank >= 0, f'invalid receiver {receiver}'

        if left_side_rank < 0 and left_side == "":
            # default left_side party is receiver party
            left_side_rank = receiver_rank
        assert left_side_rank >= 0, f'invalid receiver {left_side}'

        config = spu.psi.PsiExecuteConfig(
            protocol_conf=spu.psi.PsiProtocolConfig(
                protocol=spu.psi.parse_protocol(protocol),
                receiver_rank=receiver_rank,
                bucket_size=bucket_size,
                broadcast_result=broadcast_result,
                ecdh_params=spu.psi.EcdhParams(
                    curve=spu.psi.parse_curve_type(ecdh_curve),
                ),
            ),
            input_params=spu.psi.InputParams(
                type=spu.psi.SourceType.SOURCE_TYPE_FILE_CSV,
                path=input_path,
                selected_keys=keys,
                keys_unique=not table_keys_duplicated,
            ),
            output_params=spu.psi.OutputParams(
                type=spu.psi.SourceType.SOURCE_TYPE_FILE_CSV,
                path=output_path,
                disable_alignment=disable_alignment,
                csv_null_rep=output_csv_na_rep,
            ),
            join_conf=spu.psi.ResultJoinConfig(
                type=spu.psi.parse_join_type(join_type),
                left_side_rank=left_side_rank,
            ),
        )

        report = spu.psi.psi_execute(config, self.link)

        return {
            'party': self.party,
            'original_count': report.original_count,
            'intersection_count': report.intersection_count,
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


def _generate_input_uuid():
    return f'input-{uuid.uuid4()}'


def _generate_output_uuid():
    return f'output-{uuid.uuid4()}'


_spu_compile_lock = Lock()


def _spu_compile(fn, copts, fn_name, *meta_args, **meta_kwargs):
    meta_args, meta_kwargs = jax.tree_util.tree_map(
        lambda x: get_obj_ref(x),
        (meta_args, meta_kwargs),
    )

    # prepare inputs and metadata.
    input_name = []
    input_vis = []

    def _get_input_metadata(obj: SPUObject):
        input_name.append(_generate_input_uuid())
        input_vis.append(obj.vtype)

    jax.tree_util.tree_map(_get_input_metadata, (meta_args, meta_kwargs))

    try:
        global _spu_compile_lock
        # The current version of cachetools used by spu compile is not thread-safe
        with _spu_compile_lock:
            executable, output_tree = spu_fe.compile(
                spu_fe.Kind.JAX,
                fn,
                meta_args,
                meta_kwargs,
                input_name,
                input_vis,
                lambda output_flat: [
                    _generate_output_uuid() for _ in range(len(output_flat))
                ],
                static_argnums=(),
                static_argnames=None,
                copts=copts,
            )
    except Exception as e:
        raise RuntimeError(f"{e}")

    executable.name = fn_name
    return executable, output_tree


class SPU(Device):
    def __init__(
        self,
        cluster_def: Dict,
        link_desc: Dict = None,
        log_options: spu.logging.LogOptions = spu.logging.LogOptions(),
        id: str = None,
    ):
        """SPU device constructor.

        Args:
            cluster_def: SPU cluster definition. More details refer to
                `SPU runtime config <https://www.secretflow.org.cn/docs/spu/en/reference/runtime_config.html>`_.

                For example

                .. code:: python

                    {
                        'nodes': [
                            {
                                'party': 'alice',
                                # The address for other peers.
                                'address': '127.0.0.1:9001',
                                # The listen address of this node.
                                # Optional. Address will be used if listen_address is empty.
                                'listen_address': '',
                                # Optional. TLS related options.
                                'tls_opts': {
                                    'server_ssl_opts': {
                                        'certificate_path': 'servercert.pem',
                                        'private_key_path': 'serverkey.pem',
                                        # The options used for verify peer's client certificate
                                        'ca_file_path': 'cacert.pem',
                                        # Maximum depth of the certificate chain for verification
                                        'verify_depth': 1
                                    },
                                    'client_ssl_opts': {
                                        'certificate_path': 'clientcert.pem',
                                        'private_key_path': 'clientkey.pem',
                                        # The options used for verify peer's server certificate
                                        'ca_file_path': 'cacert.pem',
                                        # Maximum depth of the certificate chain for verification
                                        'verify_depth': 1
                                    }
                                }
                            },
                            {
                                'party': 'bob',
                                'address': '127.0.0.1:9002',
                                'listen_address': '',
                                'tls_opts': {
                                    'server_ssl_opts': {
                                        'certificate_path': "bob's servercert.pem",
                                        'private_key_path': "bob's serverkey.pem",
                                        'ca_file_path': "other's client cacert.pem",
                                        'verify_depth': 1
                                    },
                                    'client_ssl_opts': {
                                        'certificate_path': "bob's clientcert.pem",
                                        'private_key_path': "bob's clientkey.pem",
                                        'ca_file_path': "other's server cacert.pem",
                                        'verify_depth': 1
                                    }
                                }
                            },
                        ],
                        'runtime_config': {
                            'protocol': spu.ProtocolKind.SEMI2K,
                            'field': spu.FieldType.FM128,
                            'sigmoid_mode': spu.RuntimeConfig.SigmoidMode.SIGMOID_REAL,
                        }
                    }
            link_desc: Optional. A dict specifies the link parameters.
                Available parameters are:
                    1. connect_retry_times

                    2. connect_retry_interval_ms

                    3. recv_timeout_ms

                    4. http_max_payload_size

                    5. http_timeout_ms

                    6. throttle_window_size

                    7. brpc_channel_protocol refer to `https://github.com/apache/brpc/blob/master/docs/en/client.md#protocols`

                    8. brpc_channel_connection_type refer to `https://github.com/apache/brpc/blob/master/docs/en/client.md#connection-type`
            log_options: Optional. Options of spu logging.
        """
        super().__init__(DeviceType.SPU)
        self.cluster_def = cluster_def
        self.cluster_def['nodes'].sort(key=lambda x: x['party'])
        self.link_desc = link_desc
        self.log_options = log_options
        conf = spu.RuntimeConfig()
        conf.ParseFromJsonString(json.dumps(cluster_def['runtime_config']))
        # self.conf = json_format.Parse(
        #     json.dumps(cluster_def['runtime_config']), spu.RuntimeConfig()
        # )
        self.conf = conf
        self.world_size = len(self.cluster_def['nodes'])
        self.actors = {}
        self._task_id = -1
        self.io = SPUIO(self.conf, self.world_size)
        self.id = id
        self.init()

    def init(self):
        """Init SPU runtime in each party"""
        for rank, node in enumerate(self.cluster_def['nodes']):
            self.actors[node['party']] = (
                sfd.remote(SPURuntime)
                .party(node['party'])
                .remote(
                    rank,
                    self.cluster_def,
                    self.link_desc,
                    self.log_options,
                    self.id,
                )
            )

    def reset(self):
        """Reset spu to clear corrupted internal state, for test only"""
        self.shutdown()
        time.sleep(0.5)
        self.init()

    def shutdown(self):
        for actor in self.actors.values():
            sfd.kill(actor)

    def _place_arguments(self, *args, **kwargs):
        def place(obj):
            if isinstance(obj, DeviceObject):
                return obj.to(self)
            else:
                # if obj is not a DeviceObject, it should be a plaintext from
                # host program, so it's safe to mark it as VIS_PUBLIC.
                meta, io_info, *refs = self.io.make_shares(
                    obj, spu.Visibility.VIS_PUBLIC
                )

                return SPUObject(self, meta, self.infeed_shares(io_info, refs))

        return jax.tree_util.tree_map(place, (args, kwargs))

    def dump(self, obj: SPUObject, paths: List[Union[str, Callable]]):
        assert obj.device == self, "obj must be owned by this device."
        ret = []
        for i, actor in enumerate(self.actors.values()):
            ret.append(actor.dump.remote(obj.meta, obj.shares_name[i], paths[i]))
        sfd.get(ret)

    def load(self, paths: List[Union[str, Callable]]) -> SPUObject:
        outputs = [None] * self.world_size
        for i, actor in enumerate(self.actors.values()):
            actor_out = actor.load.options(num_returns=2).remote(paths[i])

            outputs[i] = actor_out

        return SPUObject(
            self, outputs[0][0], [outputs[i][1] for i in range(self.world_size)]
        )

    def __call__(
        self,
        func: Callable,
        *,
        static_argnames: Union[str, Iterable[str], None] = None,
        num_returns_policy: SPUCompilerNumReturnsPolicy = SPUCompilerNumReturnsPolicy.SINGLE,
        user_specified_num_returns: int = 1,
        copts: spu.CompilerOptions = spu.CompilerOptions(),
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

            def compile_fn(*args, **kwargs):
                return _spu_compile(*args, **kwargs)

            fn_name = get_fn_code_name(func)
            compile_fn.__name__ = f"spu_compile({fn_name})"

            # it's ok to choose any party to compile,
            # here we choose party 0.
            executable, out_shape = (
                sfd.remote(compile_fn)
                .party(self.cluster_def['nodes'][0]['party'])
                .options(num_returns=2)
                .remote(fn, copts, fn_name, *meta_args, **meta_kwargs)
            )

            if num_returns_policy == SPUCompilerNumReturnsPolicy.FROM_COMPILER:
                # Since user choose to use num of returns from compiler result,
                # the compiler result must be revealed to host.
                # Performance may hurt here.
                # However, since we only expose executable here, it's still
                # safe.
                executable, out_shape = sfd.get([executable, out_shape])
                num_returns = len(executable.output_names)

            if num_returns_policy == SPUCompilerNumReturnsPolicy.SINGLE:
                num_returns = 1

            # run executable and get returns.
            outputs = [None] * self.world_size
            for i, actor in enumerate(self.actors.values()):
                (actor_args, actor_kwargs) = jax.tree_util.tree_map(
                    lambda x: x.shares_name[i], (args, kwargs)
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
                    if len(all_atomic_spu_objects) == 1:
                        return all_atomic_spu_objects[0]
                    else:
                        return all_atomic_spu_objects

                _, out_tree = jax.tree_util.tree_flatten(out_shape)
                return jax.tree_util.tree_unflatten(out_tree, all_atomic_spu_objects)

        return wrapper

    def infeed_shares(
        self,
        io_info: FED_OBJECT_TYPES,
        shares_chunk: List[FED_OBJECT_TYPES],
    ) -> List[FED_OBJECT_TYPES]:
        assert (
            len(shares_chunk) % len(self.actors) == 0
        ), f"{len(shares_chunk)} , {len(self.actors)}"
        chunks_pre_actor = int(len(shares_chunk) / len(self.actors))

        ret = []
        for i, actor in enumerate(self.actors.values()):
            start_pos = i * chunks_pre_actor
            end_pos = (i + 1) * chunks_pre_actor
            ret.append(
                actor.infeed_share.remote(io_info, *shares_chunk[start_pos:end_pos])
            )

        return ret

    def outfeed_shares(
        self, shares_name: List[FED_OBJECT_TYPES]
    ) -> Tuple[FED_OBJECT_TYPES, List[FED_OBJECT_TYPES]]:
        assert len(shares_name) == len(self.actors)

        shares_chunk_count = sfd.get(
            (next(iter(self.actors.values()))).outfeed_shares_chunk_count.remote(
                shares_name[0]
            )
        )

        ret = []
        for i, actor in enumerate(self.actors.values()):
            remote_ret = actor.outfeed_share.options(
                num_returns=1 + shares_chunk_count
            ).remote(shares_name[i])

            if shares_chunk_count == 0:
                io_info = remote_ret
            else:
                io_info, *shares_chunk = remote_ret
                ret.extend(shares_chunk)

        return io_info, ret

    def psi_df(
        self,
        key: Union[str, List[str], Dict[Device, List[str]]],
        dfs: List[PYUObject],
        receiver: str,
        protocol='PROTOCOL_RR22',
        precheck_input=True,
        sort=True,
        broadcast_result=True,
        bucket_size=1 << 20,
        curve_type="CURVE_25519",
        dppsi_bob_sub_sampling=0.9,
        dppsi_epsilon=3,
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
            curve_type (str): curve for ecdh psi.
            preprocess_path (str): preprocess file path for unbalanced psi.
            ecdh_secret_key_path (str): ecdh_oprf secretkey file path, binary format, 32B, for unbalanced psi.
            dppsi_bob_sub_sampling (double): bob subsampling bernoulli_distribution
                probability of dp psi
            dppsi_epsilon (double): epsilon of dp psi
            progress_callbacks (Callable[[str, ProgressData], None]): Callbacks for update progress
            callbacks_interval_ms (int): The interval at which the callbacks is called

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
            dppsi_bob_sub_sampling,
            dppsi_epsilon,
        )

    def psi(
        self,
        keys: Dict[str, List[str]],
        input_path: Dict[str, str],
        output_path: Dict[str, str],
        receiver: str,
        table_keys_duplicated: Dict[str, bool] = None,
        output_csv_na_rep: str = "NULL",
        broadcast_result: bool = True,
        protocol: str = 'PROTOCOL_RR22',
        ecdh_curve: str = 'CURVE_FOURQ',
        advanced_join_type: str = "JOIN_TYPE_INNER_JOIN",
        left_side: str = "",
        disable_alignment: bool = False,
        bucket_size=1 << 20,
        dppsi_bob_sub_sampling=0.9,
        dppsi_epsilon=3,
    ):
        """Private set intersection API.
        Please check https://www.secretflow.org.cn/docs/psi/latest/en-US/reference/psi_v2_config for details.

        Args:
            keys (Dict[str, List[str]]): Keys for intersection from both parties.
            input_path (Dict[str, str]): Input paths from both parties.
            output_path (Dict[str, str]): Output paths from both parties.
            receiver (str): Name of receiver party.
            table_keys_duplicated (str): Whether keys columns catain duplicated rows. Defaults to False.
            output_csv_na_rep (str): null repsentation in output csv.
            broadcast_result (bool, optional): Whether to reveal result to sender. Defaults to True.
            protocol (str, optional): PSI protocol. Defaults to 'PROTOCOL_RR22'. Allowed values: 'PROTOCOL_ECDH', 'PROTOCOL_KKRT', 'PROTOCOL_RR22', 'PROTOCOL_ECDH_3PC', 'PROTOCOL_ECDH_NPC', 'PROTOCOL_KKRT_NPC', 'PROTOCOL_DP'
            ecdh_curve (str, optional): Curve for ECDH protocol. Only valid if ECDH is selected. Defaults to 'CURVE_FOURQ'. Allowed values: 'CURVE_25519', 'CURVE_FOURQ', 'CURVE_SM2', 'CURVE_SECP256K1'
            advanced_join_type (str, optional): Only valid if `protocol` is 'PROTOCOL_ECDH', 'PROTOCOL_KKRT' or 'PROTOCOL_RR22', Advanced Join allow duplicate keys. Defaults to "JOIN_TYPE_UNSPECIFIED". Allowed values: 'JOIN_TYPE_UNSPECIFIED', 'JOIN_TYPE_INNER_JOIN', 'JOIN_TYPE_LEFT_JOIN', 'JOIN_TYPE_RIGHT_JOIN', 'JOIN_TYPE_FULL_JOIN', 'JOIN_TYPE_DIFFERENCE'
            left_side (str, optional): Name of left join party. Required if advanced_join_type is selected. Default empty means same as receiver.
            disable_alignment (bool, optional): If true, output is not promised to be aligned. Defaults to False.

        Returns:
            Dict: PSI report.
        """

        return dispatch(
            'psi',
            self,
            keys,
            input_path,
            output_path,
            receiver,
            table_keys_duplicated,
            output_csv_na_rep,
            broadcast_result,
            protocol,
            ecdh_curve,
            advanced_join_type,
            left_side,
            disable_alignment,
            bucket_size,
            dppsi_bob_sub_sampling,
            dppsi_epsilon,
        )

    def ub_psi(
        self,
        mode: str,
        role: Dict[str, str],
        cache_path: Dict[str, str],
        input_path: Dict[str, str] = {},
        server_secret_key_path: str = '',
        keys: Dict[str, List[str]] = None,
        server_get_result: bool = False,
        client_get_result: bool = False,
        disable_alignment: bool = False,
        output_path: Dict[str, str] = {},
        join_type: str = "JOIN_TYPE_INNER_JOIN",
        left_side: str = "",
        null_rep: str = "NULL",
    ):
        """Unbalanced PSI.
        Args:
            mode (str): Mode of psi. One of [
                MODE_UNSPECIFIED,
                MODE_OFFLINE_GEN_CACHE,
                MODE_OFFLINE_TRANSFER_CACHE,
                MODE_OFFLINE,
                MODE_ONLINE,
                MODE_FULL
            ]
            role (str): Role of psi. one of [
                ROLE_SERVER,
                ROLE_CLIENT,
            ]
            input_path (str): Input path of psi.
            keys (List[str]): Keys of psi.
            server_secret_key_path (str): Server secret key path of psi.
            cache_path (str): Cache path of psi.
            server_get_result (bool): Server get result of psi.
            client_get_result (bool): Client get result of psi.
            disable_alignment (bool): Disable alignment of psi.
            output_path (str): Output path of psi.
            advanced_join_type (str, optional): Advanced Join allow duplicate keys. Defaults to "JOIN_TYPE_UNSPECIFIED". Allowed values: 'JOIN_TYPE_UNSPECIFIED', 'JOIN_TYPE_INNER_JOIN', 'JOIN_TYPE_LEFT_JOIN', 'JOIN_TYPE_RIGHT_JOIN', 'JOIN_TYPE_FULL_JOIN', 'JOIN_TYPE_DIFFERENCE'
            left_side (str, optional): Name of left join party. Required if advanced_join_type is selected. Default empty means same as receiver.
        Returns:
            Dict: PSI report output by SPU.
        """

        roles = set()
        for r in role.values():
            roles.add(r)

        return dispatch(
            'ub_psi',
            self,
            mode=mode,
            role=role,
            input_path=input_path,
            output_path=output_path,
            keys=keys,
            server_secret_key_path=server_secret_key_path,
            cache_path=cache_path,
            server_get_result=server_get_result,
            client_get_result=client_get_result,
            disable_alignment=disable_alignment,
            join_type=join_type,
            left_side=left_side,
            null_rep=null_rep,
        )
