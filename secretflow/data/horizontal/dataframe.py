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

from dataclasses import dataclass, field
from typing import Dict, Union

import numpy as np
import pandas as pd

from secretflow.data.base import DataFrameBase, Partition
from secretflow.data.ndarray import FedNdarray
from secretflow.device.device import PYU, reveal
from secretflow.security.aggregation.aggregator import Aggregator
from secretflow.security.compare.comparator import Comparator


# TODO @zhouaihui 补齐注释


@dataclass
class HDataFrame(DataFrameBase):
    """水平DataFrame"""
    partitions: Dict[PYU, Partition] = field(default_factory=dict)
    aggregator: Aggregator = None
    comparator: Comparator = None

    def mean(self, *args, **kwargs) -> pd.Series:
        assert self.aggregator is not None, 'Aggregator should be provided for mean.'
        means = [part.mean(*args, **kwargs) for part in self.partitions.values()]
        if 'numeric_only' in kwargs:
            numeric_only = kwargs['numeric_only']
        cnts = [part.count(numeric_only=numeric_only) for part in self.partitions.values()]
        return pd.Series(
            self.aggregator.average([m.values for m in means], axis=0,
                                    weights=[cnt.values for cnt in cnts]),
            index=means[0].index, dtype=np.float64)

    # TODO @zhouaihui：min/max等是否public reveal，还需要进一步讨论。
    # 如果返回HDataFrame的形式，则需要明确结果存放在哪里。如果存放在上帝视角，则上帝和device的关系我们还没讨论清楚。
    # 其他public reveal均类似。
    @reveal
    def min(self, *args, **kwargs) -> pd.Series:
        assert self.comparator is not None, 'Compartor should be provided for min.'
        mins = [part.min(*args, **kwargs) for part in self.partitions.values()]
        return pd.Series(self.comparator.min([m.values for m in mins],
                                             axis=0),
                         index=mins[0].index, dtype=np.float64)

    @reveal
    def max(self, *args, **kwargs) -> pd.Series:
        assert self.comparator is not None, 'Compartor should be provided for min.'
        maxs = [part.max(*args, **kwargs) for part in self.partitions.values()]
        return pd.Series(self.comparator.max([m.values for m in maxs],
                                             axis=0),
                         index=maxs[0].index, dtype=np.float64)

    def count(self, *args, **kwargs) -> pd.Series:
        assert self.aggregator is not None, 'Aggregator should be provided for count.'
        cnts = [part.count(*args, **kwargs) for part in self.partitions.values()]
        return pd.Series(
            self.aggregator.sum([cnt.values for cnt in cnts],
                                axis=0),
            index=cnts[0].index, dtype=np.int64)

    @property
    def values(self) -> FedNdarray:
        return FedNdarray(partitions={pyu: part.values for pyu, part in self.partitions.items()})

    @property
    def dtypes(self) -> pd.Series:
        assert len(self.partitions) > 0, 'Partitions in the dataframe is None or empty.'
        return list(self.partitions.values())[0].dtypes

    @property
    def columns(self):
        assert len(self.partitions) > 0, 'Partitions in the dataframe is None or empty.'
        return list(self.partitions.values())[0].columns

    def copy(self) -> 'HDataFrame':
        return HDataFrame(
            partitions={pyu: part.copy() for pyu, part in self.partitions.items()},
            aggregator=self.aggregator, comparator=self.comparator)

    def drop(self, labels=None, axis=0, index=None, columns=None,
             level=None, inplace=False, errors='raise') -> Union['HDataFrame', None]:
        if inplace:
            for part in self.partitions.values():
                part.drop(labels=labels, axis=axis, index=index, columns=columns,
                          level=level, inplace=inplace, errors=errors)
        else:
            return HDataFrame(
                partitions={pyu: part.drop(
                    labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace,
                    errors=errors) for pyu, part in self.partitions.items()},
                aggregator=self.aggregator, comparator=self.comparator)

    def fillna(self, value=None, method=None, axis=None,
               inplace=False, limit=None, downcast=None) -> Union['HDataFrame', None]:
        if inplace:
            for part in self.partitions.values():
                part.fillna(value=value, method=method, axis=axis, inplace=inplace, limit=limit, downcast=downcast)
        else:
            return HDataFrame(
                partitions={pyu: part.fillna(
                    value=value, method=method, axis=axis, inplace=inplace, limit=limit, downcast=downcast)
                    for pyu, part in self.partitions.items()},
                aggregator=self.aggregator, comparator=self.comparator)

    def __len__(self):
        return sum([len(part) for part in self.partitions.values()])

    def __getitem__(self, item) -> 'HDataFrame':
        return HDataFrame(aggregator=self.aggregator, comparator=self.comparator,
                          partitions={pyu: part[item] for pyu, part in self.partitions.items()})

    def __setitem__(self, key, value):
        if isinstance(value, HDataFrame):
            assert len(value.partitions) == len(
                self.partitions), f'Length of HDataFrame to assign not equals to this dataframe: {len(value.partitions)} != {len(self.partitions)}'
            for pyu, part in value.partitions.items():
                self.partitions[pyu][key] = part
        elif isinstance(value, Partition):
            assert value.data.device in self.partitions, f'Partition to assgin is not in this dataframe pyu list.'
            self.partitions[value.data.device][key] = value
        else:
            for part in self.partitions.values():
                part[key] = value
