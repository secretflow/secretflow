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
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Tuple, List, Union, Iterable, Dict

import jax
import numpy as np
import pandas as pd
import ppu
import ppu.binding._lib.libs as libs
import ppu.binding._lib.link as link
import ray
from google.protobuf import json_format
from ppu.binding import Runtime, Visibility
from ppu.ppu_pb2 import IrProto, XlaMeta

from .base import Device, DeviceType, DeviceObject
from .register import dispatch


def fxp_precision(field_type):
    """Fixed point integer default precision bits"""
    if field_type == ppu.ppu_pb2.FM32:
        return 8
    elif field_type == ppu.ppu_pb2.FM64:
        return 18
    elif field_type == ppu.ppu_pb2.FM128:
        return 26
    raise ValueError(f'unsupported field type {field_type}')


def fxp_size(field_type):
    """Fixed point integer size in bytes"""
    if field_type == ppu.ppu_pb2.FM32:
        return 4
    elif field_type == ppu.ppu_pb2.FM64:
        return 8
    elif field_type == ppu.ppu_pb2.FM128:
        return 16
    raise ValueError(f'unsupported field type {field_type}')


def argnames_partial_except(fn, static_argnames, kwargs):
    if static_argnames is None:
        return fn, kwargs

    assert isinstance(static_argnames, (str, Iterable))
    if isinstance(static_argnames, str):
        static_argnames = (static_argnames,)

    static_kwargs = {k: kwargs.pop(k) for k in static_argnames if k in kwargs}
    return functools.partial(fn, **static_kwargs), kwargs


@dataclass
class PyTreeLeaf:
    """Jax pytree leaf

    Attributes:
        name: 数组名称
        vis: 数组可见性
        dtype: 数组类型
        shape: 数组形状
    """
    name: str
    vis: ppu.binding.Visibility
    dtype: np.dtype
    shape: Tuple


class PPUObject(DeviceObject):
    def __init__(self, device: Device, data, vis='secret'):
        super().__init__(device)
        self.data = data
        self.visibility = vis


@ray.remote
class PPURuntime:
    def __init__(self, rank, cluster_def):
        self.rank = rank
        self.cluster_def = cluster_def

        desc = link.Desc()
        for node in cluster_def['nodes']:
            desc.add_party(node['id'], node['address'])
        self.conf = json_format.Parse(
            json.dumps(cluster_def['runtime_config']), ppu.RuntimeConfig())

        self.link = link.create_brpc(desc, rank)
        self.runtime = Runtime(self.link, self.conf)

    def set_var(self, names: Union[str, List[str]], *vars):
        """批量设置PPU符号表

        Args:
            names: 符号名，可以是单个变量名或者变量名列表
            *vars: 符号值

        """
        if isinstance(names, str):
            self.runtime.set_var(names, vars[0])
        else:
            [self.runtime.set_var(name, var) for name, var in zip(names, vars)]

    def get_var(self, names: Union[str, List[str]]):
        """批量获取PPU符号表

        Args:
            names: 符号名，可以是单个变量名或者变量名列表

        Returns:
            符号值
        """
        if isinstance(names, str):
            return self.runtime.get_var(names)
        else:
            return [self.runtime.get_var(name) for name in names]

    def run(self, executable):
        self.runtime.run(executable)

    def a2h(self, name, public_key=None):
        value = self.runtime.get_var(name)
        shape = value.shape.dims
        size = fxp_size(self.conf.field)
        value = np.array([
            int.from_bytes(value.content[i * size:(i + 1) * size],
                           sys.byteorder,
                           signed=True)
            for i in range(len(value.content) // size)
        ]).reshape(*shape)

        if public_key is None:
            return value
        return np.array([public_key.encrypt(x)
                         for x in value.flatten()]).reshape(*shape)

    def psi_df(self,
               key: Union[str, Iterable[str]],
               data: pd.DataFrame,
               protocol='kkrt',
               sort=True):
        """DataFrame隐私求交

        Args:
            key: 用于求交的字段，可以是一个或多个字段
            data: 待对齐的DataFrame
            protocol: PSI协议，目前支持ecdh, kkrt
            sort: 是否根据对齐后的字段对数据排序

        Returns:
            对齐后的DataFrame
        """
        assert protocol in ('ecdh',
                            'kkrt'), f'unsupported psi protocol {protocol}'

        # save key dataframe to temp file for streaming psi
        data_dir = f'.data/{self.rank}'
        os.makedirs(data_dir, exist_ok=True)
        input_path, output_path = f'{data_dir}/psi-input.csv', f'{data_dir}/psi-output.csv'
        data.to_csv(input_path, index=False)

        try:
            self.psi_csv(key, input_path, output_path, protocol, sort)

            # load result dataframe from temp file
            return pd.read_csv(output_path)
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def psi_csv(self,
                key: Union[str, Iterable[str]],
                input_path: str,
                output_path: str,
                protocol='kkrt',
                sort=True):
        """CSV文件隐私求交

        Args:
            key: 用于求交的字段，可以是一个或多个字段
            input_path: 待对齐的csv文件，逗号分割，带表头
            output_path: 对齐后的csv文件，逗号分割，带表头
            protocol: PSI协议，目前支持ecdh, kkrt
            sort: 是否根据对齐后的字段对数据排序

        Returns:
            隐私求交report
        """
        assert protocol in ('ecdh',
                            'kkrt'), f'unsupported psi protocol {protocol}'

        key = [key] if isinstance(key, str) else list(key)

        report = libs.PsiReport()
        if len(self.cluster_def['nodes']) == 2:
            if protocol == 'ecdh':
                libs.ecdh_2pc_psi(self.link, key, input_path, output_path, 1,
                                  sort, report)
            else:
                libs.kkrt_2pc_psi(self.link, key, input_path, output_path,
                                  sort, report)
        elif len(self.cluster_def['nodes']) == 3:
            assert protocol == 'ecdh', f'only ecdh-3pc psi supported'
            libs.ecdh_3pc_psi(self.link, key, input_path, output_path, sort,
                              report)
        else:
            raise ValueError('only 2pc or 3pc psi supported')

        party = self.cluster_def['nodes'][self.rank]['party']
        return {
            'party': party,
            'original_count': report.original_count,
            'intersection_count': report.intersection_count
        }


class PPU(Device):
    def __init__(self, cluster_def):
        """初始化PPU Device

        Args:
            cluster_def: PPU集群配置, 示例：
            ```python
            {
                'nodes': [
                    {
                        'party': 'alice',
                        'id': 'local:0',
                        'address': '127.0.0.1:9001'
                    },
                    {
                        'party': 'bob',
                        'id': 'local:1',
                        'address': '127.0.0.1:9002'
                    },
                ],
                'runtime_config': {
                    'protocol': ppu.ppu_pb2.SEMI2K,
                    'field': ppu.ppu_pb2.FM128,
                    'sigmoid_mode': ppu.ppu_pb2.REAL,
                }
            }
            ```
        """
        super().__init__(DeviceType.PPU)

        self.cluster_def = cluster_def
        self.conf = json_format.Parse(
            json.dumps(cluster_def['runtime_config']), ppu.RuntimeConfig())
        self.actors = {}
        self.init()

    def init(self):
        for rank, node in enumerate(self.cluster_def['nodes']):
            self.actors[node['party']] = PPURuntime.options(resources={
                node['party']: 1
            }).remote(rank, self.cluster_def)

    def reset(self):
        """reset ppu to clear corrupted internal state, for test only"""
        for actor in self.actors.values():
            ray.kill(actor)
        time.sleep(0.5)
        self.init()

    def psi_df(self,
               key: Union[str, Iterable[str]],
               dfs: List['PYUObject'],
               protocol='kkrt',
               sort=True):
        """DataFrame隐私求交

        Args:
            key: 用于求交的字段，可以是一个或多个字段
            dfs: 待对齐的DataFrame
            protocol: PSI协议，目前支持ecdh, kkrt
            sort: 是否根据对齐后的字段对数据排序

        Returns:
            对齐后的DataFrame
        """
        return dispatch('psi_df', self, key, dfs, protocol, sort)

    def psi_csv(self,
                key: Union[str, Iterable[str]],
                input_path: Union[str, Dict[Device, str]],
                output_path: Union[str, Dict[Device, str]],
                protocol='kkrt',
                sort=True):
        """CSV文件隐私求交

        Args:
            key: 用于求交的字段，可以是一个或多个字段
            input_path: 待对齐的csv文件，逗号分割，带表头
            output_path: 对齐后的csv文件，逗号分割，带表头
            protocol: PSI协议，目前支持ecdh, kkrt
            sort: 是否根据对齐后的字段对数据排序
        """
        return dispatch('psi_csv', self, key, input_path, output_path,
                        protocol, sort)

    @classmethod
    def infeed(cls, cluster_def, name, data, vis):
        """将明文数据分片(additive sharing)输入PPU Runtime

        Args:
            cluster_def: 见`PPU.__init__`
            name: 变量名
            data: 变量
            vis: 目标device对该对象的访问权限
                secret: 目标设备密文可见
                public: 目标设备明文可见

        Returns:
            返回数据分片列表和变量的PyTree
        """
        from ppu.binding import Io

        world_size = len(cluster_def['nodes'])
        runtime_config = json_format.Parse(
            json.dumps(cluster_def['runtime_config']), ppu.RuntimeConfig())
        io = Io(world_size, runtime_config)

        value_flat, value_tree = jax.tree_util.tree_flatten(data)
        value_leaves = []

        shares = {}
        for i, value in enumerate(value_flat):
            if isinstance(value, PPUObject):
                value_leaves.append(value)
                continue

            assert isinstance(value, (int, float, np.ndarray)), f'value must be int, float or np.ndarray, ' \
                                                                f'got {type(value)} instead.'

            value = value if isinstance(value, np.ndarray) else np.array(value)
            leaf = PyTreeLeaf(f'{name}_{i}', vis, value.dtype, value.shape)
            value_leaves.append(leaf)
            shares[leaf.name] = [
                ray.put(share) for share in io.make_shares(value, vis)
            ]

        return shares, jax.tree_util.tree_unflatten(value_tree, value_leaves)

    def __call__(self, func, *, static_argnames=None):
        """Set up ``fn`` for scheduling to this device.

        Args:
            func: Function to be jitted and schedule to this device.
            static_argnames: See ``jax.jit()`` docstring.

        Returns:
            A wrapped version of ``fn``, set up for just-in-time compilation and device placement.
        """

        def wrapper(*args, **kwargs):
            fn, kwargs = argnames_partial_except(func, static_argnames, kwargs)

            # step 1: feed constant args into PPU runtime
            shares, (args, kwargs) = self.infeed(self.cluster_def,
                                                 str(id(args)), (args, kwargs),
                                                 vis=Visibility.VIS_PUBLIC)
            for name, share in shares.items():
                [
                    actor.set_var.remote(name, s)
                    for actor, s in zip(self.actors.values(), share)
                ]

            # step 2: device object type check and unwrap
            value_flat, value_tree = jax.tree_util.tree_flatten((args, kwargs))
            for i, value in enumerate(value_flat):
                if isinstance(value, PPUObject):
                    assert value.device == self, f'unexpected device object {value}'
                    value_flat[i] = value.data
            args, kwargs = jax.tree_util.tree_unflatten(value_tree, value_flat)

            # step 3: collect XLA computation input visibility and names
            value_flat, _ = jax.tree_util.tree_flatten((args, kwargs))
            input_names, input_vis = [], []
            for value in value_flat:
                input_names.append(value.name)
                input_vis.append(value.vis)

            # step 4: produce XLA computation given example args
            cfn, output = jax.xla_computation(fn, return_shape=True)(*args,
                                                                     **kwargs)
            output_flat, output_tree = jax.tree_util.tree_flatten(output)

            output_leaf = [
                PyTreeLeaf(str(id(value)), Visibility.VIS_SECRET, value.dtype,
                           value.shape) for value in output_flat
            ]
            output_names = [leaf.name for leaf in output_leaf]

            ir = IrProto(ir_type=ppu.ppu_pb2.IrType.IR_XLA_HLO,
                         code=cfn.as_serialized_hlo_module_proto(),
                         meta=XlaMeta(inputs=input_vis))
            nir = ppu.binding.compile(ir)
            name = fn.func.__name__ if isinstance(
                fn, functools.partial) else fn.__name__
            executable = ppu.ppu_pb2.ExecutableProto(name=name,
                                                     input_names=input_names,
                                                     output_names=output_names,
                                                     code=nir.code)

            # step 5: execute jitted function in PPU
            [actor.run.remote(executable) for actor in self.actors.values()]

            return_value = jax.tree_util.tree_unflatten(
                output_tree, output_leaf)
            if isinstance(return_value, Tuple):
                return [PPUObject(self, value) for value in return_value]
            else:
                return PPUObject(self, return_value)

        return wrapper
