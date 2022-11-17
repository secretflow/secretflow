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

from typing import Dict

from secretflow.data.base import Partition
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.io import read_csv_wrapper
from secretflow.device import PYU
from secretflow.security.aggregation.aggregator import Aggregator
from secretflow.security.compare.comparator import Comparator


def read_csv(
    filepath: Dict[PYU, str],
    aggregator: Aggregator = None,
    comparator: Comparator = None,
    **kwargs,
) -> HDataFrame:
    """Read a comma-separated values (csv) file into HDataFrame.

    Args:
        filepath: a dict {PYU: file path}.
        aggregator: optionla; the aggregator assigned to the dataframe.
        comparator: optionla; the comparator assigned to the dataframe.
        kwargs: all other arguments are same with :py:meth:`pandas.DataFrame.read_csv`.

    Returns:
        HDataFrame

    Examples:
        >>> read_csv({PYU('alice'): 'alice.csv', PYU('bob'): 'bob.csv'})
    """
    assert filepath, 'File path shall not be empty!'
    df = HDataFrame(aggregator=aggregator, comparator=comparator)
    for device, path in filepath.items():
        df.partitions[device] = Partition(device(read_csv_wrapper)(path, **kwargs))
    # Check column and dtype.
    dtypes = None
    for part in df.partitions.values():
        if dtypes is None:
            dtypes = part.dtypes
        else:
            dtypes_next = part.dtypes
            assert dtypes.equals(
                dtypes_next
            ), f'Different dtypes: {dtypes} vs {dtypes_next}'

    return df


def to_csv(df: HDataFrame, file_uris: Dict[PYU, str], **kwargs):
    """Write object to a comma-separated values (csv) file.

    Args:
        df: the HDataFrame to save.
        file_uris: the file path of each PYU.
        kwargs: all other arguments are same with :py:meth:`pandas.DataFrame.to_csv`.
    """
    return [
        df.partitions[device].to_csv(uri, **kwargs) for device, uri in file_uris.items()
    ]
