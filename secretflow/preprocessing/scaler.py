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

from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

from secretflow.data.base import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.mix import MixDataFrame
from secretflow.data.vertical import VDataFrame


class MinMaxScaler:
    @staticmethod
    def _check_dataframe(df):
        assert isinstance(df, (HDataFrame, VDataFrame, MixDataFrame)
                          ), f'Accepts HDataFrame/VDataFrame/MixDataFrame only but got {type(df)}'

    def fit(self, df: Union[HDataFrame, VDataFrame, MixDataFrame]):
        self._check_dataframe(df)
        min_max = pd.concat([df.min().to_frame(name='min').transpose(), df.max().to_frame(name='max').transpose()])
        self._scaler = SkMinMaxScaler()
        self._scaler.fit(min_max)

    def _transform(self, df: Union[HDataFrame, VDataFrame]) -> Union[HDataFrame, VDataFrame]:
        transformed_parts = {}

        def _df_transform(scaler: SkMinMaxScaler, df: pd.DataFrame):
            new_df = df.copy()
            new_df.iloc[:, :] = scaler.transform(df)
            return new_df
        for device, part in df.partitions.items():
            scaler = SkMinMaxScaler()
            mask = np.in1d(self._scaler.feature_names_in_, part.dtypes.index, assume_unique=True)
            scaler.fit(np.stack([self._scaler.data_min_[mask], self._scaler.data_max_[mask]]))
            transformed_parts[device] = Partition(device(_df_transform)(scaler, part.data))

        new_df = df.copy()
        new_df.partitions = transformed_parts
        return new_df

    def transform(self, df: Union[HDataFrame, VDataFrame, MixDataFrame]) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        assert hasattr(self, '_scaler'), 'Scaler has not been fit yet.'
        self._check_dataframe(df)
        if isinstance(df, (HDataFrame, VDataFrame)):
            return self._transform(df)
        else:
            return MixDataFrame(partitions=[self._transform(part) for part in df.partitions])

    def fit_transform(self, df: Union[HDataFrame, VDataFrame]):
        self.fit(df)
        return self.transform(df)
