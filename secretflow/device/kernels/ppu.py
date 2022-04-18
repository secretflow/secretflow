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

from typing import List, Union, Iterable, Dict

import jax
import ray
from ppu.binding import Io

from secretflow.device.device import PYU, PYUObject, HEU, HEUObject, PPU, PPUObject
from secretflow.device.device import register, DeviceType, Device


def _outfeed(conf, pytree, *vars):
    io = Io(len(vars), conf)
    value_flat, value_tree = jax.tree_util.tree_flatten(pytree)

    value_leaves = []
    for i in range(len(value_flat)):
        value_leaves.append(io.reconstruct([var[i] for var in vars]))

    return jax.tree_util.tree_unflatten(value_tree, value_leaves)


@register(DeviceType.PPU)
def to(self: PPUObject, device: Device, vis):
    if isinstance(device, PYU):
        value_flat, value_tree = jax.tree_util.tree_flatten(self.data)
        names = [value.name for value in value_flat]
        vars = [actor.get_var.remote(names) for actor in self.device.actors.values()]
        return device(_outfeed)(self.device.conf, self.data, *vars)
    elif isinstance(device, HEU):
        assert len(self.device.actors) == 2, f'Only support 2PC PPU-HEU conversion'
        parties = self.device.actors.keys()
        assert device.generator in parties and device.evaluator in parties, f'Mismatch PPU and HEU parties'

        # TODO(@xibin.wxb): support pytree
        s1 = self.device.actors[device.generator].a2h.remote(self.data.name, public_key=device.pk)
        s2 = self.device.actors[device.evaluator].a2h.remote(self.data.name)
        data = device.a2h.options(resources={device.evaluator: 1}).remote(s1, s2)
        return HEUObject(device, data)

    raise ValueError(f'Unexpected device type: {type(device)}')


def _check_psi_protocol(device: PPU, protocol):
    assert protocol in ('kkrt', 'ecdh'), f'unknown psi protocol {protocol}'
    assert len(device.actors) == 2 or len(device.actors) == 3, f'only 2pc or 3pc psi supported'
    if len(device.actors) == 3:
        assert protocol == 'ecdh', f'only ecdh-3pc psi supported'


@register(DeviceType.PPU)
def psi_df(device: PPU, key: Union[str, Iterable[str]],
           dfs: List['PYUObject'], protocol='kkrt', sort=True) -> List[PYUObject]:
    """DataFrame隐私求交

    Examples:

    ```python
    ppu = sf.PPU(utils.cluster_def)

    df1, df2 = Partition(), Partition()
    df1, df2 = ppu.psi_df('uid', [df1, df2])
    ```

    Args:
        device: PPU设备
        key: 用于求交的字段，可以是一个或多个字段
        dfs: 待对齐的DataFrame
        protocol: PSI协议，目前支持ecdh, kkrt
        sort: 是否根据对齐后的字段对数据排序

    Returns:
        对齐后的DataFrame。例如
        df1 = pd.DataFrame({'key': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
                            'A': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3']})

        df2 = pd.DataFrame({'key': ['K3', 'K1', 'K9', 'K4'],
                            'A': ['A3', 'A1', 'A9', 'A4']})

        df1, df2 = ppu.psi_df('key', [df1, df2])

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
    assert isinstance(device, PPU), f'device must be PPU device'
    assert isinstance(key, (str, Iterable)), f'invalid key, must be str or list of str'
    assert len(set([df.device for df in dfs])) == len(dfs), f'dataframe should not be in same PYU device'
    assert len(dfs) == len(device.actors), f'unexpected number of dataframes, should be {len(device.actors)}'
    _check_psi_protocol(device, protocol)

    for df in dfs:
        assert isinstance(df, PYUObject), f'dataframe must be in PYU device'
        assert df.device.party in device.actors, f'{df.device} not co-located with {device}'

    res = []
    for df in dfs:
        actor = device.actors[df.device.party]
        res.append(PYUObject(df.device, actor.psi_df.remote(key, df.data, protocol, sort)))

    return res


@register(DeviceType.PPU)
def psi_csv(device: PPU, key: Union[str, Iterable[str]], input_path: Union[str, Dict[Device, str]],
            output_path: Union[str, Dict[Device, str]], protocol='kkrt', sort=True):
    """CSV文件隐私求交

    Examples:

    ```python
    ppu = sf.PPU(utils.cluster_def)
    alice = sf.PYU('alice'), sf.PYU('bob')

    input_path = {alice: '/path/to/alice.csv', bob: '/path/to/bob.csv'}
    output_path = {alice: '/path/to/alice_psi.csv', bob: '/path/to/bob_psi.csv'}

    ppu.psi_csv(['c1', 'c2'], input_path, output_path)
    ```

    Args:
        device: PPU设备
        key: 用于求交的字段，可以是一个或多个字段
        input_path: 待对齐的csv文件，逗号分割，带表头
        output_path: 对齐后的csv文件，逗号分割，带表头
        protocol: PSI协议，目前支持ecdh, kkrt
        sort: 是否根据对齐后的字段对数据排序

    Returns:
        List: 各方的隐私求交report
    """
    assert isinstance(device, PPU), f'device must be PPU device'
    assert isinstance(key, (str, Iterable)), f'invalid key, must be str or list of str'
    assert isinstance(input_path, (str, Dict)), f'input_path must be str or dict of str'
    assert isinstance(output_path, (str, Dict)), f'input_path must be str or dict of str'
    _check_psi_protocol(device, protocol)

    if isinstance(input_path, Dict):
        assert input_path.keys() == output_path.keys(), f'mismatch input_path and out_path devices'
        assert len(input_path) == len(device.actors), f'unexpected number of dataframes, should be {len(device.actors)}'

        for dev in input_path.keys():
            assert dev.party in device.actors, f'input_path {dev} not co-located with {device}'

        for dev in output_path.keys():
            assert dev.party in device.actors, f'output_path {dev} not co-located with {device}'

    res = []
    if isinstance(input_path, str):
        assert isinstance(output_path, str), f'input_path and output_path must be same types'
        for actor in device.actors:
            res.append(actor.psi_csv.remote(key, input_path, output_path, protocol, sort))
    else:
        for dev, ipath in input_path.items():
            opath = output_path[dev]
            actor = device.actors[dev.party]
            res.append(actor.psi_csv.remote(key, ipath, opath, protocol, sort))

    # wait for all tasks done
    return ray.get(res)
