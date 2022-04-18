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

from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd
from pandas.core.indexes.base import Index

from secretflow.data.base import DataFrameBase, Partition
from secretflow.data.ndarray import FedNdarray
from secretflow.device.device import PYU, reveal
from secretflow.device.device.base import Device
from secretflow.utils.errors import NotFoundError

# TODO @zhouaihui: 补齐注释


@dataclass
class VDataFrame(DataFrameBase):
    """垂直DataFrame"""
    partitions: Dict[PYU, Partition]
    aligned: bool = True

    def min(self, *args, **kwargs) -> pd.Series:
        return pd.concat(reveal([part.min(*args, **kwargs).data for part in self.partitions.values()]))

    def max(self, *args, **kwargs) -> pd.Series:
        return pd.concat(reveal([part.max(*args, **kwargs).data for part in self.partitions.values()]))

    @property
    def dtypes(self) -> pd.Series:
        return pd.concat([part.dtypes for part in self.partitions.values()])

    @property
    def columns(self):
        assert len(self.partitions) > 0, 'Partitions in the dataframe is None or empty.'
        cols = None
        for part in self.partitions.values():
            if cols is None:
                cols = part.columns
            else:
                cols = cols.append(part.columns)
        return cols

    def mean(self, *args, **kwargs) -> pd.Series:
        return pd.concat(reveal([part.mean(*args, **kwargs).data for part in self.partitions.values()]))

    def count(self, *args, **kwargs) -> pd.Series:
        return pd.concat(reveal([part.count(*args, **kwargs).data for part in self.partitions.values()]))

    @property
    def values(self):
        return FedNdarray(partitions={pyu: part.values for pyu, part in self.partitions.items()})

    def copy(self) -> 'VDataFrame':
        return VDataFrame(partitions={pyu: part.copy() for pyu, part in self.partitions.items()}, aligned=self.aligned)

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise') -> Union[
            'VDataFrame', None]:
        if columns:
            col_index = self._col_index(columns)
            if inplace:
                for pyu, col in col_index.items():
                    self.partitions[pyu].drop(labels=labels, axis=axis, index=index,
                                              columns=col, level=level, inplace=inplace, errors=errors)
            else:
                new_parts = self.partitions.copy()
                for pyu, col in col_index.items():
                    new_parts[pyu] = self.partitions[pyu].drop(
                        labels=labels, axis=axis, index=index, columns=col, level=level, inplace=inplace, errors=errors)
                return VDataFrame(partitions=new_parts, aligned=self.aligned)

        else:
            if inplace:
                for part in self.partitions.values():
                    part.drop(labels=labels, axis=axis, index=index,
                              columns=columns, level=level, inplace=inplace, errors=errors)
            else:
                return VDataFrame(
                    partitions={pyu: part.drop(
                        labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace,
                        errors=errors) for pyu, part in self.partitions.items()},
                    aligned=self.aligned)

    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None) -> Union['VDataFrame', None]:
        if inplace:
            for part in self.partitions.values():
                part.fillna(value=value, method=method, axis=axis, inplace=inplace, limit=limit, downcast=downcast)
        else:
            return VDataFrame(
                partitions={pyu: part.fillna(
                    value=value, method=method, axis=axis, inplace=inplace, limit=limit, downcast=downcast)
                    for pyu, part in self.partitions.items()},
                aligned=self.aligned)

    def __len__(self):
        """若未对齐，则返回分块数据中的最大长度。
        """
        parts = list(self.partitions.values())
        assert parts, 'No partitions in vdataframe.'
        return max([len(part) for part in parts])

    def _col_index(self, col) -> Dict[Device, Union[str, List[str]]]:
        assert col.tolist() if isinstance(col, Index) else col, f'Column to index is None or empty!'
        pyu_col = {}
        listed_col = col.tolist() if isinstance(col, Index) else col
        if not isinstance(listed_col, (list, tuple)):
            listed_col = [listed_col]
        for key in listed_col:
            found = False
            for pyu, part in self.partitions.items():
                if key not in part.dtypes:
                    continue

                found = True
                if pyu not in pyu_col:
                    pyu_col[pyu] = key
                else:
                    if not isinstance(pyu_col[pyu], list):
                        # 有多个列，则转为列表。
                        pyu_col[pyu] = [pyu_col[pyu]]
                    pyu_col[pyu].append(key)

                break

            if not found:
                raise NotFoundError(f'Item {key} does not exist.')
        return pyu_col

    def __getitem__(self, item) -> 'VDataFrame':
        item_index = self._col_index(item)
        return VDataFrame(
            partitions={pyu: self.partitions[pyu][keys] for pyu, keys in item_index.items()})

    def __setitem__(self, key, value):
        if isinstance(value, Partition):
            assert value.data.device in self.partitions, 'Device of the partition to assgin is not in this dataframe devices.'
            self.partitions[value.data.device][key] = value
            return
        elif isinstance(value, VDataFrame):
            for pyu in value.partitions.keys():
                assert pyu in self.partitions, 'Partitions to assgin is not same with this dataframe partitions.'
            try:
                key_index = self._col_index(key)
                for pyu, col in key_index.items():
                    self.partitions[pyu][col] = value.partitions[pyu] if isinstance(value, VDataFrame) else value
            except NotFoundError:
                # Key没有出现过，则视作新增列。
                # 注意，此时所有的Partition都会被视为新增的列。
                for pyu, part in value.partitions.items():
                    self.partitions[pyu][part.dtypes.index] = part
        else:
            key_index = self._col_index(key)
            for pyu, col in key_index.items():
                self.partitions[pyu][col] = value.partitions[pyu] if isinstance(value, VDataFrame) else value
