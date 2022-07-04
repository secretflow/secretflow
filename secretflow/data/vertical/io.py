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
from secretflow.device import SPU, PYU


def read_csv(
    filepath: Dict[PYU, str],
    delimiter=',',
    dtypes: Dict[PYU, Dict[str, type]] = None,
    spu: SPU = None,
    keys: Union[str, List[str]] = None,
    drop_keys=False,
) -> VDataFrame:
    """Read a comma-separated values (csv) file into VDataFrame.

    When specifying spu and keys, the fields specified by keys are used for PSI
    alignment.Fields used for alignment must be common to all parties, and other
    fields cannot be repeated across parties. The data for each party is
    supposed pre-aligned if not specifying spu and keys.

    Args:
        filepath: The file path of each party. It can be a local file with a
            relative or absolute path, or a remote file starting with `oss://`,
            `http(s)://`, E.g.

            .. code:: python

                {
                    PYU('alice'): 'oss://bucket/data/alice.csv',
                    PYU('bob'): 'oss://bucket/data/bob.csv'
                }
        delimiter: the file separator.
        dtypes: Participant field type. It will be inferred from the file if
            not specified, E.g.

            .. code:: python

                {
                    PYU('alice'): {'uid': np.str, 'age': np.int32},
                    PYU('bob'): {'uid': np.str, 'score': np.float32}
                }
        spu: SPU device, used for PSI data alignment.
            The data of all parties are supposed pre-aligned if not specified.
        keys: The field used for psi, which can be single or multiple fields.
            This parameter is required when spu is specified.
        drop_keys: whether to remove keys.

    Returns:
        A aligned VDataFrame.
    """
    assert len(filepath) == 2, f'only support 2 parties for now'
    assert spu is None or keys is not None, f'keys required when spu provided'

    partitions = {}

    for device, path in filepath.items():
        usecols = dtypes[device].keys() if dtypes is not None else None
        dtype = dtypes[device] if dtypes is not None else None
        partitions[device] = Partition(
            device(read_csv_wrapper)(
                path, delimiter=delimiter, usecols=usecols, dtype=dtype
            )
        )

    # TODO(@xibin.wxb): use psi_csv instead of psi_df
    if spu is not None:
        dfs = spu.psi_df(keys, [part.data for part in partitions.values()])
        partitions = {df.device: Partition(df) for df in dfs}

    if drop_keys:
        for device, partition in partitions.items():
            assert keys is not None, f"Cannot find keys={keys} when doing drop keys"
            if isinstance(keys, str):
                assert (
                    keys in partition.columns
                ), f"Cannot find keys={keys} when doing drop keys"
            elif isinstance(keys, List):
                columns_set = set(partition.columns)
                keys_set = set(keys)
                assert columns_set.issuperset(
                    keys_set
                ), f"keys = {keys_set.difference(columns_set)} can not find on device {device}"
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


def to_csv(df: VDataFrame, file_uris: Dict[PYU, str], **kwargs):
    """Write object to a comma-separated values (csv) file.

    Args:
        df: the VDataFrame to save.
        file_uris: the file path of each PYU.
        kwargs: all other arguments are same with :py:meth:`pandas.DataFrame.to_csv`.
    """
    for device, uri in file_uris.items():
        df.partitions[device].to_csv(uri, **kwargs)
