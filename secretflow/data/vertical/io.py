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

from typing import Dict, List, Union

from secretflow.data.base import Partition
from secretflow.data.io import read_csv_wrapper
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import PPU, PYU


def read_csv(filepath: Dict[PYU, str], delimiter=',', dtypes: Dict[PYU, Dict[str, type]] = None,
             ppu: PPU = None, keys: Union[str, List[str]] = None, drop_keys=False) -> VDataFrame:
    """创建垂直DataFrame

    当指定ppu和keys时，使用keys指定的字段用于PSI对齐；否则，默认各方数据已经预对齐。用于对齐的字段必须各方公有，其他字段不能在各方重复。

    Args:
        filepath: 参与方文件地址，地址可以是相对或绝对路径的本地文件，或者以`oss://`, `http(s)://`开头的远程文件，例如：
          {PYU('alice'): 'oss://bucket/data/alice.csv', PYU('bob'): 'oss://bucket/data/bob.csv'}
        delimiter: 文件分割符
        dtypes: 参与方字段类型，若不指定，根据各方文件进行推断。例如：
          {
             PYU('alice'): {
               'uid': np.str,
               'age': np.int32,
             },
             PYU('bob'): {
               'uid': np.str,
               'score': np.float32,
             }
          }
        ppu: PPU设备，用于PSI数据对齐；若不指定，则默认数据预对齐
        keys: 用于对齐的字段，可以是单个或者多个字段；当指定ppu时，该参数必需
        drop_keys: 是否删除用于对齐的字段

    Returns:
        对齐后的垂直DataFrame
    """
    assert len(filepath) == 2, f'only support 2 parties for now'
    assert ppu is None or keys is not None, f'keys required when ppu provided'

    partitions = {}

    for device, path in filepath.items():
        usecols = dtypes[device].keys() if dtypes is not None else None
        dtype = dtypes[device] if dtypes is not None else None
        partitions[device] = Partition(device(read_csv_wrapper)(
            path, delimiter=delimiter, usecols=usecols, dtype=dtype))

    # TODO(@xibin.wxb): use psi_csv instead of psi_df
    if ppu is not None:
        dfs = ppu.psi_df(keys, [part.data for part in partitions.values()])
        partitions = {df.device: Partition(df) for df in dfs}

    if drop_keys:
        for device, partition in partitions.items():
            assert keys is not None, f"Cannot find keys={keys} when doing drop keys"
            if isinstance(keys, str):
                assert keys in partition.columns, f"Cannot find keys={keys} when doing drop keys"
            elif isinstance(keys, List):
                columns_set = set(partition.columns)
                keys_set = set(keys)
                assert columns_set.issuperset(
                    keys_set), f"keys = {keys_set.difference(columns_set)} can not find on device {device}"
            else:
                raise Exception(f"Illegal type for keys,got {type(keys)}")
            partitions[device] = partition.drop(labels=keys, axis=1)
        keys = None

    unique_cols = set()
    length = None
    for device, partition in partitions.items():
        n = len(partition)
        dtype = partition.dtypes
        if keys is not None:
            dtype = dtype.drop(labels=keys)

        if length is None:
            length = n
        else:
            assert length == n, f'number of samples must be equal across all devices'

        # data columns must be unique across all devices
        for col in dtype.index:
            assert col not in unique_cols, f'col {col} duplicate in multiple devices'
            unique_cols.add(col)
    return VDataFrame(partitions)
