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

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from secretflow.data import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.mix import MixDataFrame, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device.driver import reveal
from secretflow.preprocessing.base import _PreprocessBase
from secretflow.security.aggregation import Aggregator
from secretflow.utils.errors import InvalidArgumentError


class _STDScaler:
    """
    Standard scaler for pd.DataFrame, which is used to replace sklearn.preprocessing.StandardScaler
    """

    def __init__(self, *, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X):
        if self.with_mean:
            self.mean_ = X.mean()
        else:
            self.mean_ = None
        if self.with_std:
            self.var_ = X.var(ddof=0)
            self.scale_ = X.std(ddof=0)
        else:
            self.var_ = None
            self.scale_ = None

    def transform(self, X: pd.DataFrame):
        if not hasattr(self, 'mean_') or not hasattr(self, 'scale_'):
            raise RuntimeError("You must fit the scaler before calling transform.")
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, 'mean_') or not hasattr(self, 'scale_'):
            raise RuntimeError(
                "You must fit the scaler before calling inverse_transform."
            )
        if self.with_std:
            X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X


class StandardScaler(_PreprocessBase):
    """Standardize features by removing the mean and scaling to unit variance.

    StandardScaler is similar to :py:class:`sklearn.preprocessing.StandardScaler`.
    The main differences are
    a) takes HDataFrame/VDataFrame/MixDataFrame as input/output.
    b) does not support sparse matrix.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.

    Attributes:
        _scaler : the sklearn StandardScaler instance.
        _with_mean : bool, default=True
            if True, center the data before scaling.
        _with_std : bool, default=True
            If True, scale the data to unit variance (or equivalently,
            unit standard deviation).

    Examples:
        >>> from secretflow.preprocessing import StandardScaler
        >>> data = HDataFrame(...) # your HDataFrame/VDataFrame/MixDataFrame instance.
        >>> scaler = StandardScaler()
        >>> scaler.fit(data)
        >>> print(scaler._scaler.mean_, scaler._scaler.var_)
        >>> scaler.transform(data)
    """

    def __init__(self, with_mean=True, with_std=True) -> None:
        """
        Args:
            with_mean: optional; same as sklearn StandardScaler。
            with_std: optional; same as sklearn StandardScaler
        """
        self._with_mean = with_mean
        self._with_std = with_std

    @staticmethod
    def _check_dataframe(df):
        assert isinstance(
            df, (HDataFrame, VDataFrame, MixDataFrame)
        ), f'Accepts HDataFrame/VDataFrame/MixDataFrame only but got {type(df)}'

    def _fit_horizontal(
        self, partitions: List[Partition], aggregator: Aggregator = None
    ) -> _STDScaler:
        means = [part.mean(numeric_only=True) for part in partitions]
        cnts = [part.count(numeric_only=True) for part in partitions]
        mean = reveal(
            aggregator.average(
                [m.values for m in means], axis=0, weights=[cnt.values for cnt in cnts]
            )
        )
        count = reveal(aggregator.sum([cnt.values for cnt in cnts], axis=0))
        scaler = _STDScaler(with_mean=self._with_mean, with_std=self._with_std)
        scaler.mean_ = mean if self._with_mean else None

        if self._with_std:

            def _cal_var(_mean, df: pd.DataFrame):
                return ((df - _mean) ** 2).sum()

            vars = [part.data.device(_cal_var)(mean, part.data) for part in partitions]
            scaler.var_ = reveal(aggregator.sum(vars, axis=0))
            if not scaler.var_.shape:
                scaler.var_ = np.array([scaler.var_])
            scaler.var_ = scaler.var_ / count
            scaler.scale_ = np.sqrt(scaler.var_)

        return scaler

    def _concatenate_scaler(self, scalers: List[_STDScaler]) -> _STDScaler:
        scaler = _STDScaler(with_mean=self._with_mean, with_std=self._with_std)
        if self._with_mean:
            scaler.mean_ = np.concatenate([scaler.mean_ for scaler in scalers])
        else:
            scaler.mean_ = None
        if self._with_std:
            scaler.scale_ = np.concatenate([scaler.scale_ for scaler in scalers])
            scaler.var_ = np.concatenate([scaler.var_ for scaler in scalers])
        else:
            scaler.scale_ = None
            scaler.var_ = None
        return scaler

    def fit(
        self,
        df: Union[HDataFrame, VDataFrame, MixDataFrame],
        aggregator: Aggregator = None,
    ):
        """Fit a federated dataframe.

        Args:
            df: the X to fit.
            aggregator: optional; the aggregator to compute global mean
                and standard variance. Shall provided if X is a horizontal
                partitioned MixDataFrame.
        """
        self._check_dataframe(df)
        self._columns = df.columns
        if isinstance(df, MixDataFrame):
            if df.partition_way == PartitionWay.HORIZONTAL:
                if self._with_mean or self._with_std:
                    assert aggregator is not None, (
                        'Should provide a aggregator for horizontal partitioned'
                        'MixDataFrame when with_mean or with_std is true'
                    )

                parts_list = [list(part.partitions.values()) for part in df.partitions]
                scalers = [
                    self._fit_horizontal(parts, aggregator)
                    for parts in zip(*parts_list)
                ]
            else:
                scalers = [
                    self._fit_horizontal(
                        list(hdf.partitions.values()),
                        aggregator if aggregator is not None else hdf.aggregator,
                    )
                    for hdf in df.partitions
                ]
            self._scaler = self._concatenate_scaler(scalers)
        elif isinstance(df, HDataFrame):
            self._scaler = self._fit_horizontal(
                list(df.partitions.values()),
                aggregator if aggregator is not None else df.aggregator,
            )
        else:
            # VDataFrame
            def _sk_fit(with_mean, with_std, _df: pd.DataFrame):
                scaler = _STDScaler(with_mean=with_mean, with_std=with_std)
                scaler.fit(_df)
                return scaler

            scalers = [
                reveal(device(_sk_fit)(self._with_mean, self._with_std, part.data))
                for device, part in df.partitions.items()
            ]
            self._scaler = self._concatenate_scaler(scalers)

    def _transform(
        self, scaler: _STDScaler, df: Union[HDataFrame, VDataFrame]
    ) -> Union[HDataFrame, VDataFrame]:
        def _df_transform(_df: pd.DataFrame, _scaler: _STDScaler):
            _new_df = _df.copy()
            _new_df.iloc[:, :] = _scaler.transform(_df)
            return _new_df

        transformed_parts = {}
        if isinstance(df, HDataFrame):
            for device, part in df.partitions.items():
                transformed_parts[device] = part.apply_func(
                    _df_transform, _scaler=scaler
                )
        elif isinstance(df, VDataFrame):
            start_idx = 0
            end_idx = 0
            for device, part in df.partitions.items():
                end_idx += len(part.columns)
                part_scaler = _STDScaler(
                    with_mean=self._with_mean, with_std=self._with_std
                )
                if self._with_mean:
                    part_scaler.mean_ = scaler.mean_[start_idx:end_idx]
                else:
                    part_scaler.mean_ = None
                if self._with_std:
                    part_scaler.var_ = scaler.var_[start_idx:end_idx]
                    part_scaler.scale_ = scaler.scale_[start_idx:end_idx]
                else:
                    part_scaler.var_ = None
                    part_scaler.scale_ = None
                transformed_parts[device] = part.apply_func(
                    _df_transform, _scaler=part_scaler
                )
                start_idx = end_idx
        else:
            raise InvalidArgumentError(
                f'_transform accepts HDataFrame/VDataFrame only but got {type(df)}'
            )

        new_df = df.copy()
        new_df.partitions = transformed_parts
        return new_df

    def transform(
        self, df: Union[HDataFrame, VDataFrame, MixDataFrame]
    ) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        """Transform a federated dataframe.

        Args:
            df: the X to transform.

        Returns:
            a federated dataframe corresponding to the input X.
        """
        # Sanity check.
        assert hasattr(self, '_scaler'), 'Scaler has not been fit yet.'
        self._check_dataframe(df)
        features_num = None
        if self._with_mean:
            features_num = len(self._scaler.mean_)
        if self._with_std:
            features_num = len(self._scaler.var_)
        if features_num is not None:
            assert len(df.columns) == features_num, (
                f'X has {len(df.columns)} features, but StandardScaler '
                f'is expecting {features_num} features as input.'
            )

        if isinstance(df, (HDataFrame, VDataFrame)):
            return self._transform(self._scaler, df)

        new_parts = []
        if df.partition_way == PartitionWay.HORIZONTAL:
            for part in df.partitions:
                new_parts.append(self._transform(self._scaler, part))
            return MixDataFrame(partitions=new_parts)
        else:
            start_idx = 0
            end_idx = 0
            for part in df.partitions:
                end_idx += len(part.columns)
                part_scaler = _STDScaler(
                    with_mean=self._with_mean, with_std=self._with_std
                )
                part_scaler.mean_ = self._scaler.mean_[start_idx:end_idx]
                part_scaler.var_ = self._scaler.var_[start_idx:end_idx]
                part_scaler.scale_ = self._scaler.scale_[start_idx:end_idx]
                new_parts.append(self._transform(part_scaler, part))
                start_idx = end_idx
        return MixDataFrame(partitions=new_parts)

    def fit_transform(
        self, df: Union[HDataFrame, VDataFrame], aggregator: Aggregator = None
    ):
        """A convenience combine of fit and transform."""
        self.fit(df, aggregator=aggregator)
        return self.transform(df)

    def get_params(self) -> Dict[str, Any]:
        assert hasattr(self, '_scaler'), 'Scaler has not been fit yet.'

        return {
            'columns': self._columns,
            'with_mean': self._scaler.with_mean,
            'mean': self._scaler.mean_,
            'with_std': self._scaler.with_std,
            'scale': self._scaler.scale_,
        }
