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
from secretflow.device import PYU, SPU, Device
from secretflow.utils.errors import InvalidArgumentError
from secretflow.utils.random import global_random


def read_csv(
    filepath: Dict[PYU, str],
    delimiter=',',
    dtypes: Dict[PYU, Dict[str, type]] = None,
    spu: SPU = None,
    keys: Union[str, List[str], Dict[Device, List[str]]] = None,
    drop_keys: Union[str, List[str], Dict[Device, List[str]]] = None,
    psi_protocl=None,
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
                    PYU('alice'): 'alice.csv',
                    PYU('bob'): 'bob.csv'
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
        drop_keys: keys to removed, which can be single or multiple fields.
            This parameter is required when spu is specified since VDataFrame
            doesn't allow duplicate column names.
        psi_protocl: Specified protocol for PSI. Default 'KKRT_PSI_2PC' for 2
            parties, 'ECDH_PSI_3PC' for 3 parties.

    Returns:
        A aligned VDataFrame.
    """
    assert spu is None or keys is not None, f'keys required when spu provided'
    assert spu is None or drop_keys is not None, f'drop_keys required when spu provided'
    if spu is not None:
        assert len(filepath) <= 3, f'only support 2 or 3 parties for now'

    def get_keys(
        device: Device, x: Union[str, List[str], Dict[Device, List[str]]] = None
    ) -> List[str]:
        if x:
            if isinstance(x, str):
                return [x]
            elif isinstance(x, List):
                return x
            elif isinstance(x, Dict):
                if device in x:
                    if isinstance(x[device], str):
                        return [x[device]]
                    else:
                        return x[device]
            else:
                raise InvalidArgumentError(f"Illegal type for keys,got {type(x)}")
        else:
            return []

    filepath_actual = filepath
    if spu is not None:
        if psi_protocl is None:
            psi_protocl = 'KKRT_PSI_2PC' if len(filepath) == 2 else 'ECDH_PSI_3PC'
        rand_suffix = global_random(list(filepath.keys())[0], 100000)
        output_file = {
            pyu: f'{path}.psi_output_{rand_suffix}' for pyu, path in filepath.items()
        }
        spu.psi_csv(
            keys,
            input_path=filepath,
            output_path=output_file,
            protocol=psi_protocl,
            receiver=list(filepath.keys())[0].party,
        )
        filepath_actual = output_file

    partitions = {}
    for device, path in filepath_actual.items():
        usecols = dtypes[device].keys() if dtypes is not None else None
        dtype = dtypes[device] if dtypes is not None else None
        partitions[device] = Partition(
            device(read_csv_wrapper)(
                path, delimiter=delimiter, usecols=usecols, dtype=dtype
            )
        )

    if drop_keys:
        for device, partition in partitions.items():
            device_drop_key = get_keys(device, drop_keys)
            device_psi_key = get_keys(device, keys)

            if device_drop_key is not None:
                columns_set = set(partition.columns)
                device_drop_key_set = set(device_drop_key)
                assert columns_set.issuperset(device_drop_key_set), (
                    f"drop_keys = {device_drop_key_set.difference(columns_set)}"
                    " can not find on device {device}"
                )

                device_psi_key_set = set(device_psi_key)
                assert device_psi_key_set.issuperset(device_drop_key_set), (
                    f"drop_keys = {device_drop_key_set.difference(device_psi_key_set)} "
                    f"can not find on device_psi_key_set of device {device},"
                    f" which are {device_psi_key_set}"
                )

                partitions[device] = partition.drop(labels=device_drop_key, axis=1)

    unique_cols = set()
    length = None

    # data columns must be unique across all devices
    for device, partition in partitions.items():
        n = len(partition)
        dtype = partition.dtypes

        if length is None:
            length = n
        else:
            assert length == n, f'number of samples must be equal across all devices'

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
    return [
        df.partitions[device].to_csv(uri, **kwargs) for device, uri in file_uris.items()
    ]
