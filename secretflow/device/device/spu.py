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
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import ray
from google.protobuf import json_format

import spu
import spu.binding._lib.libs as libs
import spu.binding._lib.link as link
from spu import spu_pb2
from spu import Io, Runtime
from heu import phe

from .base import Device, DeviceObject, DeviceType
from .register import dispatch
from .type_traits import spu_fxp_size, spu_datatype_to_heu
from .utils import check_num_returns, flatten


def argnames_partial_except(fn, static_argnames, kwargs):
    if static_argnames is None:
        return fn, kwargs

    assert isinstance(static_argnames, (str, Iterable))
    if isinstance(static_argnames, str):
        static_argnames = (static_argnames,)

    static_kwargs = {k: kwargs.pop(k) for k in static_argnames if k in kwargs}
    return functools.partial(fn, **static_kwargs), kwargs


@dataclass
class CustomPyTreeNode:
    """Custom pytree node wrapper for pickling.

    NOTE: For custom pytree node, e.g `jax.example_libraries.optimizers.OptimizerState`,
    it may contain `jaxlib.xla_extension.PyTreeDef` which is not pickable. So we have to
    pickle its flattend value and reconstruct it in remote.

    Attributes:
        value: Flattened value of custom node.
        builder: Callable that creates this custom node.
    """

    value: Any
    builder: Callable


@dataclass
class PyTreeLeaf:
    """SPUObject meta info.

    Attributes:
        name: object name in SPU runtime
        vis: object visibility
        dtype: object data type
        shape: object array shape
    """

    name: str
    vis: spu.Visibility
    dtype: np.dtype
    shape: Tuple


class SPUObject(DeviceObject):
    """SPU device object.

    Attributes:
        data (ray.ObjectRef): Reference to `PyTreeLeaf` which represents raw data meta info.
    """

    def __init__(self, device: 'SPU', data: ray.ObjectRef):
        super().__init__(device)
        self.data = data


@ray.remote
class SPURuntime:
    def __init__(self, rank: int, cluster_def: Dict):
        """SPURuntime constructor.

        Args:
            rank (int): Communicator rank of this SPU runtime.
            cluster_def (Dict): Cluster definition of all SPU runtimes,
            including node, address, protocol, etc.
        """
        self.rank = rank
        self.cluster_def = cluster_def

        desc = link.Desc()
        for i, node in enumerate(cluster_def['nodes']):
            if i == rank and node.get('listen_address', ''):
                desc.add_party(node['id'], node['listen_address'])
            else:
                desc.add_party(node['id'], node['address'])
        self.conf = json_format.Parse(
            json.dumps(cluster_def['runtime_config']), spu.RuntimeConfig()
        )

        self.link = link.create_brpc(desc, rank)
        self.runtime = Runtime(self.link, self.conf)

    def set_var(self, vars: Dict[str, Any]):
        """Set variables to SPU runtime.

        Args:
            vars: Dict{var_name -> ValueProto}
        """
        for name, var in vars.items():
            self.runtime.set_var(name, var)

    def get_var(self, tree: PyTreeLeaf):
        """Get variables from SPU runtime.

        Args:
            tree: Tree struct with all leaf nodes are PyTreeLeaf,
            can be PyTreeLeaf, List[PyTreeLeaf], etc.

        Returns:
            List[np.ndarray]: List of values corresponding to leaf nodes of flatten tree.
        """
        value_flat, _ = jax.tree_util.tree_flatten(tree)
        return [self.runtime.get_var(value.name) for value in value_flat]

    def run(
        self,
        task_id: int,
        fn: Callable,
        static_argnames: Union[str, Iterable[str], None],
        args: List,
        kwargs: Dict,
    ):
        """Execute function.

        Args:
            task_id (int): Task ID of this.
            fn (Callable): Function to be executed.
            static_argnames (Union[str, Iterable[str], None]): See ``jax.jit()`` docstring.
            args (List): Positional arguments, which should be either constant or tree
            struct with all leaf nodes are PyTreeLeaf.
            kwargs (Dict): Keyword arguments, which should be either constant or tree
            struct with all leaf nodes are PyTreeLeaf.

        Returns:
            Tree struct with all leaf nodes are PyTreeLeaf,
            can be PyTreeLeaf, List[PyTreeLeaf], etc.
        """
        from spu import Visibility, IrProto, XlaMeta

        fn, kwargs = argnames_partial_except(fn, static_argnames, kwargs)
        args, kwargs = flatten(args, kwargs)

        # step 1: feed constant args into SPU runtime
        value_shares = SPU.infeed(
            self.cluster_def,
            f'task_{task_id}_input',
            (args, kwargs),
            vis=Visibility.VIS_PUBLIC,
        )
        shares, (args, kwargs) = value_shares[:-1], value_shares[-1]
        self.set_var(shares[self.rank])

        # step 2: collect XLA computation input visibility and names
        value_flat, _ = jax.tree_util.tree_flatten((args, kwargs))
        input_names, input_vis = [], []
        for value in value_flat:
            input_names.append(value.name)
            input_vis.append(value.vis)

        # step 3: produce XLA computation given example args
        cfn, output = jax.xla_computation(fn, return_shape=True)(*args, **kwargs)
        output_flat, output_tree = jax.tree_util.tree_flatten(output)

        output_leaf = [
            PyTreeLeaf(
                f'task_{task_id}_output_{i}',
                Visibility.VIS_SECRET,
                value.dtype,
                value.shape,
            )
            for i, value in enumerate(output_flat)
        ]
        output_names = [leaf.name for leaf in output_leaf]

        # step 4: execute jitted function in SPU
        ir = IrProto(
            ir_type=spu.spu_pb2.IrType.IR_XLA_HLO,
            code=cfn.as_serialized_hlo_module_proto(),
            meta=XlaMeta(inputs=input_vis),
        )
        nir = spu.compile(ir)
        name = fn.func.__name__ if isinstance(fn, functools.partial) else fn.__name__
        executable = spu.spu_pb2.ExecutableProto(
            name=name,
            input_names=input_names,
            output_names=output_names,
            code=nir.code,
        )
        self.runtime.run(executable)

        return jax.tree_util.tree_unflatten(output_tree, output_leaf)

    def a2h(self, tree: PyTreeLeaf, exp_heu_data_type: str):
        """Convert SPUObject to HEUObject.

        Args:
            tree (PyTreeLeaf): SPUObject meata info.
            exp_heu_data_type (str): HEU data type.

        Returns:
            np.ndarray: Array of `phe.Plaintext`.
        """
        assert isinstance(
            tree, PyTreeLeaf
        ), f'A2H only support single array conversion.'

        value = self.runtime.get_var(tree.name)
        expect_st = f"semi2k.AShr<{spu_pb2.FieldType.Name(self.conf.field)}>"
        assert (
            value.storage_type == expect_st
        ), f"Unsupported storage type {value.storage_type}, expected {expect_st}"

        assert spu_datatype_to_heu(value.data_type) == exp_heu_data_type, (
            f"You cannot feed {value.data_type} into this HEU since it only "
            f"supports {exp_heu_data_type}, if you want to change data type of HEU, "
            f"please modify the initial configuration of HEU."
        )

        shape = value.shape.dims
        size = spu_fxp_size(self.conf.field)
        value = np.array(
            [
                # The data from SPU is already encoded.
                # convert to phe.Plaintext to avoid double encoding
                phe.Plaintext(
                    int.from_bytes(
                        value.content[i * size : (i + 1) * size],
                        sys.byteorder,
                        signed=True,
                    )
                )
                for i in range(len(value.content) // size)
            ]
        ).reshape(*shape)
        return value

    def psi_df(
        self,
        key: Union[str, Iterable[str]],
        data: pd.DataFrame,
        protocol='kkrt',
        sort=True,
    ):
        """Private set intersection with DataFrame.

        Args:
            key (Union[str, Iterable[str]]): Column(s) used to join.
            data (pd.DataFrame): DataFrame to be joined.
            protocol (str): PSI protocol, supported: ecdh, kkrt.
            sort (bool): Whether sort data by key after join.

        Returns:
            pd.DataFrame: joined DataFrame. For example:

            df1 = pd.DataFrame({'key': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
                                'A': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3']})

            df2 = pd.DataFrame({'key': ['K3', 'K1', 'K9', 'K4'],
                                'A': ['A3', 'A1', 'A9', 'A4']})

            df1, df2 = spu.psi_df('key', [df1, df2])

            >>> df1
                key   A
            1   K1    A1
            5   K3    A3
            4   K4    A4

            >>> df2
                key   A
            1   K1    A1
            0   K3    A3
            3   K4    A4
        """
        assert protocol in ('ecdh', 'kkrt'), f'unsupported psi protocol {protocol}'

        # save key dataframe to temp file for streaming psi
        data_dir = f'.data/{self.rank}'
        os.makedirs(data_dir, exist_ok=True)
        input_path, output_path = (
            f'{data_dir}/psi-input.csv',
            f'{data_dir}/psi-output.csv',
        )
        data.to_csv(input_path, index=False)

        try:
            self.psi_csv(key, input_path, output_path, protocol, sort)

            # load result dataframe from temp file
            return pd.read_csv(output_path)
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def psi_csv(
        self,
        key: Union[str, Iterable[str]],
        input_path: str,
        output_path: str,
        protocol='kkrt',
        sort=True,
    ):
        """Private set intersection with csv file.

        Examples
        --------
        >>> spu = sf.SPU(utils.cluster_def)
        >>> alice = sf.PYU('alice'), sf.PYU('bob')
        >>> input_path = {alice: '/path/to/alice.csv', bob: '/path/to/bob.csv'}
        >>> output_path = {alice: '/path/to/alice_psi.csv', bob: '/path/to/bob_psi.csv'}
        >>> spu.psi_csv(['c1', 'c2'], input_path, output_path)

        Args:
            key (Union[str, Iterable[str]]): Column(s) used to join.
            input_path: CSV file to be joined, comma seperated and contains header.
            output_path: Joined csv file, comma seperated and contains header.
            protocol (str): PSI protocol, supported: ecdh, kkrt.
            sort (bool): Whether sort data by key after join.

        Returns:
            Dict: PSI report output by SPU.
        """
        assert protocol in ('ecdh', 'kkrt'), f'unsupported psi protocol {protocol}'

        key = [key] if isinstance(key, str) else list(key)

        report = libs.PsiReport()
        if len(self.cluster_def['nodes']) == 2:
            if protocol == 'ecdh':
                libs.ecdh_2pc_psi(
                    self.link, key, input_path, output_path, 1, sort, report
                )
            else:
                libs.kkrt_2pc_psi(
                    self.link, key, input_path, output_path, sort, report, True
                )
        elif len(self.cluster_def['nodes']) == 3:
            assert protocol == 'ecdh', f'only ecdh-3pc psi supported'
            libs.ecdh_3pc_psi(self.link, key, input_path, output_path, sort, report)
        else:
            raise ValueError('only 2pc or 3pc psi supported')

        party = self.cluster_def['nodes'][self.rank]['party']
        return {
            'party': party,
            'original_count': report.original_count,
            'intersection_count': report.intersection_count,
        }


class SPU(Device):
    def __init__(self, cluster_def: Dict):
        """SPU device constructor.

        Args:
            cluster_def (Dict): SPU cluster definition. For example:
            .. code:: python

                {
                    'nodes': [
                        {
                            'party': 'alice',
                            'id': 'local:0',
                            'address': '127.0.0.1:9001',
                            'listen_address': '' # Optional. Address will be used if listen_address is empty.
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
        """
        super().__init__(DeviceType.SPU)

        self.cluster_def = cluster_def
        self.conf = json_format.Parse(
            json.dumps(cluster_def['runtime_config']), spu.RuntimeConfig()
        )
        self.actors = {}
        self._task_id = -1
        self.init()

    def init(self):
        """Init SPU runtime in each party"""
        for rank, node in enumerate(self.cluster_def['nodes']):
            self.actors[node['party']] = SPURuntime.options(
                resources={node['party']: 1}
            ).remote(rank, self.cluster_def)

    def reset(self):
        """Reset spu to clear corrupted internal state, for test only"""
        for actor in self.actors.values():
            ray.kill(actor)
        time.sleep(0.5)
        self.init()

    def get_field_type(self):
        """Get SPU fix point integer type"""
        return self.conf.field  # spu.spu_pb2.FM128

    def psi_df(
        self,
        key: Union[str, Iterable[str]],
        dfs: List['PYUObject'],
        protocol: str = 'kkrt',
        sort=True,
    ):
        """Private set intersection with DataFrame.

        Args:
            key (Union[str, Iterable[str]]): Column(s) used to join.
            dfs (List[PYUObject]): DataFrames to be joined, which
            should be colocated with SPU runtimes.
            protocol (str): PSI protocol, supported: ecdh, kkrt.
            sort (bool): Whether sort data by key after join.

        Returns:
            List[PYUObject]: Joined DataFrames with order reserved.
        """
        return dispatch('psi_df', self, key, dfs, protocol, sort)

    def psi_csv(
        self,
        key: Union[str, Iterable[str]],
        input_path: Union[str, Dict[Device, str]],
        output_path: Union[str, Dict[Device, str]],
        protocol='kkrt',
        sort=True,
    ):
        """Private set intersection with csv file.

        Args:
            key (Union[str, Iterable[str]]): Column(s) used to join.
            input_path: CSV files to be joined, comma seperated and contains header.
            output_path: Joined csv files, comma seperated and contains header.
            protocol (str): PSI protocol, supported: ecdh, kkrt.
            sort (bool): Whether sort data by key after join.

        Returns:
            List[Dict]: PSI reports output by SPU with order reserved.
        """
        return dispatch('psi_csv', self, key, input_path, output_path, protocol, sort)

    @classmethod
    def infeed(cls, cluster_def: Dict, name: str, data: Any, vis):
        """Secret share data and feed into SPU runtime.

        Args:
            cluster_def: See `SPU.__init__`.
            name: variable name.
            data: variable value.
            vis: Visibility in SPU
                secret: secret share
                public: public share

        Returns:
            Tree struct corresponding to data with all leaf nodes are PyTreeLeaf.
        """
        from spu import Io

        world_size = len(cluster_def['nodes'])
        runtime_config = json_format.Parse(
            json.dumps(cluster_def['runtime_config']), spu.RuntimeConfig()
        )
        io = Io(world_size, runtime_config)

        value_flat, value_tree = jax.tree_util.tree_flatten(data)
        value_leaves = []

        value_shares = [{} for _ in range(world_size)]
        for i, value in enumerate(value_flat):
            if isinstance(value, PyTreeLeaf):
                value_leaves.append(value)
                continue

            if isinstance(value, CustomPyTreeNode):
                _, tree = jax.tree_util.tree_flatten(value.builder())
                value_leaves.append(jax.tree_util.tree_unflatten(tree, value.value))
                continue

            assert isinstance(
                value,
                (int, float, np.ndarray, pd.DataFrame, pd.Series, jnp.DeviceArray),
            ), (
                f'value must be int, float, pd.DataFrame or np.ndarray, '
                f'got {type(value)} instead.'
            )

            # NOTE: JAX enables 32bit mode by default, so we wrap all values
            # with jnp.array to convert them to int32 or float32.
            # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

            # TODO(@xibin.wxb): remove np.array once if io.make_shares supports jnp.array.
            value = np.array(jnp.array(value))
            leaf = PyTreeLeaf(f'{name}_{i}', vis, value.dtype, value.shape)
            value_leaves.append(leaf)
            shares = io.make_shares(value, vis)
            for rank in range(world_size):
                value_shares[rank][leaf.name] = shares[rank]

        value_shares.append(jax.tree_util.tree_unflatten(value_tree, value_leaves))
        return value_shares

    @classmethod
    def outfeed(cls, conf: Dict, tree: Any, *shares):
        """Reconstruct data with secret shares.

        Args:
            conf (Dict): SPU runtime config, see `SPU.__init__`.
            tree (Any): Tree struct of original data.
            shares (Tuple[List[np.ndarray]]): List of values corresponding to leaf nodes in tree.

        Returns:
            Original data feed into SPU.
        """
        io = Io(len(shares), conf)
        value_flat, value_tree = jax.tree_util.tree_flatten(tree)

        value_leaves = []
        for i in range(len(value_flat)):
            value_leaves.append(io.reconstruct([share[i] for share in shares]))

        return jax.tree_util.tree_unflatten(value_tree, value_leaves)

    def __call__(
        self,
        func: Callable,
        *,
        num_returns: int = None,
        static_argnames: Union[str, Iterable[str], None] = None,
    ):
        """Set up ``fn`` for scheduling to this device.

        Args:
            func (Callable): Function to be jitted and schedule to this device.
            num_returns (int): Number of returned SPUObject.
            static_argnames (Union[str, Iterable[str], None]): See ``jax.jit()`` docstring.

        Returns:
            A wrapped version of ``fn``, set up for just-in-time compilation and device placement.
        """
        self._task_id += 1
        task_id = self._task_id

        def wrapper(*args, **kwargs):
            _num_returns = (
                check_num_returns(func) if num_returns is None else num_returns
            )

            # device object type check and unwrap
            value_flat, value_tree = jax.tree_util.tree_flatten((args, kwargs))
            for i, value in enumerate(value_flat):
                if isinstance(value, SPUObject):
                    assert (
                        value.device == self
                    ), f'unexpected device object {value.device} self {self}'
                    value_flat[i] = value.data
            args, kwargs = jax.tree_util.tree_unflatten(value_tree, value_flat)

            # execute jitted function in SPURuntime
            res = None
            for actor in self.actors.values():
                res = actor.run.options(num_returns=_num_returns).remote(
                    task_id, func, static_argnames, args, kwargs
                )

            if _num_returns == 1:
                return SPUObject(self, res)
            else:
                return [SPUObject(self, value) for value in res]

        return wrapper
