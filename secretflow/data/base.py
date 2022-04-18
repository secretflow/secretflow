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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import pandas as pd
from pandas.core.indexes.base import Index

from secretflow.device.device import PYUObject, reveal


class DataFrameBase(ABC):
    @abstractmethod
    def min(self):
        pass

    @abstractmethod
    def max(self):
        pass

    @abstractmethod
    def count(self):
        pass

    @abstractmethod
    def values(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass


# TODO @饶飞: 补齐注释


@dataclass
class Partition(DataFrameBase):
    """水平或者垂直数据分片，组成DataFrame的基本单元"""
    data: PYUObject = None

    def mean(self, *args, **kwargs) -> 'Partition':
        return Partition(self.data.device(pd.DataFrame.mean)(self.data, *args, **kwargs))

    def min(self, *args, **kwargs) -> 'Partition':
        return Partition(self.data.device(pd.DataFrame.min)(self.data, *args, **kwargs))

    def max(self, *args, **kwargs) -> 'Partition':
        return Partition(self.data.device(pd.DataFrame.max)(self.data, *args, **kwargs))

    def count(self, *args, **kwargs) -> 'Partition':
        return Partition(self.data.device(pd.DataFrame.count)(self.data, *args, **kwargs))

    @property
    def values(self):
        return self.data.device(lambda df: df.values)(self.data)

    @property
    @reveal
    def index(self):
        return self.data.device(lambda df: df.index)(self.data)

    @property
    @reveal
    def dtypes(self):
        # return series always.
        return self.data.device(
            lambda df: df.dtypes if isinstance(df, pd.DataFrame) else pd.Series({df.name: df.types}))(
            self.data)

    @property
    @reveal
    def columns(self):
        return self.data.device(lambda df: df.columns)(self.data)

    def iloc(self, index) -> 'Partition':
        return Partition(self.data.device(lambda df, index: df.iloc[index])(self.data, index))

    def drop(self, labels=None, axis=0, index=None, columns=None,
             level=None, inplace=False, errors='raise') -> Union['Partition', None]:
        def _drop(df: pd.DataFrame, **kwargs):
            if inplace:
                new_df = df.copy(deep=True)
                new_df.drop(**kwargs)
                return new_df
            else:
                return df.drop(**kwargs)

        new_data = self.data.device(_drop)(self.data, labels=labels, axis=axis, index=index,
                                           columns=columns, level=level, inplace=inplace, errors=errors)
        if inplace:
            self.data = new_data
        else:
            return Partition(new_data)

    def fillna(self, value=None, method=None, axis=None,
               inplace=False, limit=None, downcast=None) -> Union['Partition', None]:
        def _fillna(df: pd.DataFrame, **kwargs):
            if inplace:
                new_df = df.copy(deep=True)
                new_df.fillna(**kwargs)
                return new_df
            else:
                return df.fillna(**kwargs)

        new_data = self.data.device(_fillna)(self.data, value=value, method=method,
                                             axis=axis, inplace=inplace, limit=limit, downcast=downcast)
        if inplace:
            self.data = new_data
        else:
            return Partition(new_data)

    def rename(self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None,
               errors='ignore') -> Union['Partition', None]:
        def _rename(df: pd.DataFrame, **kwargs):
            if inplace:
                new_df = df.copy(deep=True)
                new_df.rename(**kwargs)
                return new_df
            else:
                return df.rename(**kwargs)

        new_data = self.data.device(_rename)(self.data, mapper=mapper, index=index, columns=columns,
                                             axis=axis, copy=copy, inplace=inplace, level=level, errors=errors)
        if inplace:
            self.data = new_data
        else:
            return Partition(new_data)

    def value_counts(self, *args, **kwargs) -> 'Partition':
        return Partition(self.device(pd.DataFrame.value_counts)(self.data, *args, **kwargs))

    @reveal
    def __len__(self):
        return self.data.device(lambda df: len(df))(self.data)

    def __getitem__(self, item) -> 'Partition':
        """注意和pandas.DataFrame稍有不同，即使是获取单列，返回的也是DataFrame而不是Series。
        """
        item_list = item
        if not isinstance(item, (list, tuple, Index)):
            item_list = [item_list]
        return Partition(self.data.device(pd.DataFrame.__getitem__)(self.data, item_list))

    def __setitem__(self, key, value):
        if isinstance(value, Partition):
            assert self.data.device == value.data.device, f'Can not assign a partition with different device.'

        def _setitem(df: pd.DataFrame, key, value):
            # DataFrame属于ray zero-copy的类型，其数据存放在shared memory，ray.get获取到的对象是只读的。
            # 所以这里需要深度拷贝。
            df = df.copy(deep=True)
            df[key] = value
            return df

        self.data = self.data.device(_setitem)(self.data, key,
                                               value if not isinstance(value, Partition) else value.data)

    def copy(self):
        return Partition(self.data)
