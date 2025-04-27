# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from secretflow.data.horizontal import HDataFrame
from secretflow.data.mix import MixDataFrame
from secretflow.data.vertical import VDataFrame
from secretflow.preprocessing.base import _PreprocessBase


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler


from typing import Any, Dict, Union


class MinMaxScaler(_PreprocessBase):
    """Transform features by scaling each feature to a given range.

    Attributes:
        _scaler: the sklearn MinMaxScaler instance.

    Examples:
        >>> from secretflow.preprocessing import MinMaxScaler
        >>> scaler = MinMaxScaler()
        >>> scaler.fit(df)
        >>> scaler.transform(df)
    """

    @staticmethod
    def _check_dataframe(df):
        assert isinstance(
            df, (HDataFrame, VDataFrame, MixDataFrame)
        ), f'Accepts HDataFrame/VDataFrame/MixDataFrame only but got {type(df)}'

    def fit(self, df: Union[HDataFrame, VDataFrame, MixDataFrame]):
        """Compute the minimum and maximum for later scaling."""
        self._check_dataframe(df)
        min_max = pd.concat(
            [
                df.min().to_frame(name='min').transpose(),
                df.max().to_frame(name='max').transpose(),
            ]
        )
        self._scaler = SkMinMaxScaler()
        self._scaler.fit(min_max)
        self._columns = df.columns

    def _transform(
        self, df: Union[HDataFrame, VDataFrame]
    ) -> Union[HDataFrame, VDataFrame]:
        transformed_parts = {}

        def _df_transform(_df, _scaler: SkMinMaxScaler):
            pl = None
            try:
                import polars as pl
            except ImportError:
                pass
            if isinstance(_df, pd.DataFrame):
                _new_df = _df.copy()
                _new_df.iloc[:, :] = _scaler.transform(_df)
            elif pl is not None and isinstance(_df, pl.DataFrame):
                _new_df = _df.clone()
                _new_df = pl.DataFrame(_scaler.transform(_new_df), schema=_df.columns)
            else:
                raise RuntimeError(f"Unknwon df type {type(_df)}")
            return _new_df

        for device, part in df.partitions.items():
            scaler = SkMinMaxScaler()
            mask = np.in1d(
                self._scaler.feature_names_in_, part.columns, assume_unique=True
            )
            scaler.fit(
                np.stack([self._scaler.data_min_[mask], self._scaler.data_max_[mask]])
            )
            transformed_parts[device] = part.apply_func(_df_transform, _scaler=scaler)

        new_df = df.copy()
        new_df.partitions = transformed_parts
        return new_df

    def transform(
        self, df: Union[HDataFrame, VDataFrame, MixDataFrame]
    ) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        """Scale features of X according to feature_range."""
        assert hasattr(self, '_scaler'), 'Scaler has not been fit yet.'
        self._check_dataframe(df)
        if isinstance(df, (HDataFrame, VDataFrame)):
            return self._transform(df)
        else:
            return MixDataFrame(
                partitions=[self._transform(part) for part in df.partitions]
            )

    def fit_transform(self, df: Union[HDataFrame, VDataFrame]):
        """Fit to X, then transform X."""
        self.fit(df)
        return self.transform(df)

    def get_params(self) -> Dict[str, Any]:
        assert hasattr(self, '_scaler'), 'Scaler has not been fit yet.'

        return {
            'columns': self._columns,
            'min': self._scaler.min_,
            'scale': self._scaler.scale_,
        }
