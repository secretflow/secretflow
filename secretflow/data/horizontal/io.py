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

from secretflow.device import PYU
from secretflow.security.aggregation.aggregator import Aggregator
from secretflow.security.compare.comparator import Comparator

from ..core import partition
from ..core.io import read_csv_wrapper
from .dataframe import HDataFrame


def read_csv(
    filepath: Dict[PYU, str],
    aggregator: Aggregator = None,
    comparator: Comparator = None,
    backend: str = "pandas",
    **kwargs,
) -> HDataFrame:
    """Read a comma-separated values (csv) file into HDataFrame.

    Args:
        filepath: a dict {PYU: file path}.
        aggregator: optionla; the aggregator assigned to the dataframe.
        comparator: optionla; the comparator assigned to the dataframe.
        backend: optional; the backend of the dataframe, default to 'pandas'.
        kwargs: all other arguments.

    Returns:
        HDataFrame

    Examples:
        >>> read_csv({PYU('alice'): 'alice.csv', PYU('bob'): 'bob.csv'})
    """
    assert filepath, "File path shall not be empty!"
    df = HDataFrame(aggregator=aggregator, comparator=comparator)
    for device, path in filepath.items():
        df.partitions[device] = partition(
            data=read_csv_wrapper,
            device=device,
            backend=backend,
            filepath=path,
            read_backend=backend,
            **kwargs,
        )
    # Check column and dtype.
    dtypes = None
    for part in df.partitions.values():
        if dtypes is None:
            dtypes = part.dtypes
        else:
            dtypes_next = part.dtypes
            for key in part.dtypes:
                if key not in dtypes_next or dtypes_next[key] != part.dtypes[key]:
                    raise RuntimeError(f"Different dtypes: {dtypes} vs {dtypes_next}")

    return df
