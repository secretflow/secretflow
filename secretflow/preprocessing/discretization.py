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
from sklearn.preprocessing import KBinsDiscretizer as SkKBinsDiscretizer

from secretflow.data.base import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.mix.dataframe import MixDataFrame, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device.driver import reveal
from secretflow.preprocessing.binning.homo_binning import HomoBinning
from secretflow.security.aggregation import Aggregator
from secretflow.security.compare import Comparator
from secretflow.preprocessing.base import _PreprocessBase

_STRATEGIES = ['uniform', 'quantile']


class KBinsDiscretizer(_PreprocessBase):
    """Bin continuous data into intervals.

    This KBinsDiscretizer is almost same as
    :py:class:`sklearn.preprocessing.KBinsDiscretizer`
    where the input and output are federated dataframe.

    Attributes:
        _discretizer: the sklearn.preprocessing.KBinsDiscretizer instance used.
        _n_bins: The number of bins to produce.
        _strategy: {'uniform', 'quantile'}, notice that 'kmeans' is not supported yet now.
    """

    def __init__(self, n_bins=5, strategy: str = 'quantile') -> None:
        assert (
            strategy in _STRATEGIES
        ), f'Invalid strategy {strategy}, should be one of {_STRATEGIES}'
        self._n_bins = n_bins
        self._strategy = strategy
        self._encode = 'ordinal'

    @staticmethod
    def _check_dataframe(df):
        assert isinstance(
            df, (HDataFrame, VDataFrame, MixDataFrame)
        ), f'Accepts HDataFrame/VDataFrame/MixDataFrame only but got {type(df)}'

    def _fit_hdf(
        self,
        df: HDataFrame,
        compress_thres: int = None,
        error: float = None,
        max_iter: int = None,
    ):
        assert df.aggregator is not None, 'HDataFrame should provide a aggregator.'
        binner = HomoBinning(
            bin_num=self._n_bins,
            compress_thres=compress_thres,
            error=error,
            max_iter=max_iter,
        )
        result = reveal(binner.fit_split_points(df))
        discretizer = SkKBinsDiscretizer(
            n_bins=self._n_bins, encode=self._encode, strategy=self._strategy
        )
        discretizer.bin_edges_ = np.array(list(result.values()))
        discretizer.n_bins_ = np.array([len(v) for v in result.values()])
        return discretizer

    def _concat_discretizer(self, ests: List[SkKBinsDiscretizer]) -> SkKBinsDiscretizer:
        discretizer = SkKBinsDiscretizer(
            n_bins=self._n_bins, encode=self._encode, strategy=self._strategy
        )
        discretizer.bin_edges_ = np.concatenate([est.bin_edges_ for est in ests])
        discretizer.n_bins_ = np.concatenate([est.n_bins_ for est in ests])
        return discretizer

    def fit(
        self,
        df: Union[HDataFrame, VDataFrame, MixDataFrame],
        aggregator: Aggregator = None,
        comparator: Comparator = None,
        compress_thres: int = 10000,
        error: float = 1e04,
        max_iter: int = 200,
    ) -> 'KBinsDiscretizer':
        """Fit the estimator.

        Args:
            df: the X to fit.
            aggregator: optional; shall be provided if df is a horizontal partitioned MixDataFrame.
            comparator: optional; shall be provided if df is a horizontal partitioned MixDataFrame.
            compress_thres: optional; the compress threshold of :py:class:`~secretflow.preprocessing.binning.homo_binning.HomoBinning`.
            error: optional; the error of :py:class:`~secretflow.preprocessing.binning.homo_binning.HomoBinning`.
            max_iter: optional; the max iterations of :py:class:`~secretflow.preprocessing.binning.homo_binning.HomoBinning`.

        Returns:
            the instance itself.
        """
        self._check_dataframe(df)
        self._columns = df.columns
        if self._strategy == 'uniform':
            min_max = pd.concat(
                [
                    df.min().to_frame(name='min').transpose(),
                    df.max().to_frame(name='max').transpose(),
                ]
            )
            self._discretizer = SkKBinsDiscretizer(
                n_bins=self._n_bins, encode=self._encode, strategy=self._strategy
            )
            self._discretizer.fit(min_max)
        else:
            # Quantile binning.
            if isinstance(df, HDataFrame):
                self._discretizer = self._fit_hdf(
                    df, compress_thres=compress_thres, error=error, max_iter=max_iter
                )
            elif isinstance(df, VDataFrame):

                def _sk_dis(n_bins, encode, strategy, df: pd.DataFrame):
                    discretizer = SkKBinsDiscretizer(
                        n_bins=n_bins, encode=encode, strategy=strategy
                    )
                    discretizer.fit(df)
                    return discretizer

                ests = [
                    reveal(
                        device(_sk_dis)(
                            self._n_bins, self._encode, self._strategy, part.data
                        )
                    )
                    for device, part in df.partitions.items()
                ]
                self._discretizer = self._concat_discretizer(ests)
            else:
                # MixDataFrame
                if df.partition_way == PartitionWay.HORIZONTAL:
                    assert (
                        aggregator is not None
                    ), f'Should privide a aggregator when df is a horizontal partitioned MixDataFrame.'
                    assert (
                        comparator is not None
                    ), f'Should privide a comparator when df is a horizontal partitioned MixDataFrame.'
                    hdfs = [
                        HDataFrame(aggregator=aggregator, comparator=comparator)
                        for i in range(len(df.partitions[0].partitions))
                    ]
                    for vdf in df.partitions:
                        for i, item in enumerate(vdf.partitions.items()):
                            hdfs[i].partitions[item[0]] = item[1]
                    ests = [
                        self._fit_hdf(
                            hdf,
                            compress_thres=compress_thres,
                            error=error,
                            max_iter=max_iter,
                        )
                        for hdf in hdfs
                    ]
                    self._discretizer = self._concat_discretizer(ests)
                else:
                    ests = [
                        self._fit_hdf(
                            hdf,
                            compress_thres=compress_thres,
                            error=error,
                            max_iter=max_iter,
                        )
                        for hdf in df.partitions
                    ]
                    self._discretizer = self._concat_discretizer(ests)

        return self

    def _transform(
        self, df: Union[HDataFrame, VDataFrame], est: SkKBinsDiscretizer
    ) -> Union[HDataFrame, VDataFrame]:
        transformed_parts = {}

        def _df_transform(est: SkKBinsDiscretizer, df: pd.DataFrame):
            new_df = df.copy()
            new_df.iloc[:, :] = est.transform(df)
            return new_df

        if isinstance(df, HDataFrame):
            for device, part in df.partitions.items():
                transformed_parts[device] = Partition(
                    device(_df_transform)(est, part.data)
                )
        else:
            # VDataFrame
            start_idx = 0
            end_idx = 0
            for device, part in df.partitions.items():
                end_idx += len(part.columns)
                est_part = SkKBinsDiscretizer(
                    n_bins=self._n_bins, encode=self._encode, strategy=self._strategy
                )
                est_part.bin_edges_ = est.bin_edges_[start_idx:end_idx]
                est_part.n_bins_ = est.n_bins_[start_idx:end_idx]
                transformed_parts[device] = Partition(
                    device(_df_transform)(est_part, part.data)
                )
                start_idx = end_idx

        new_df = df.copy()
        new_df.partitions = transformed_parts
        return new_df

    def transform(
        self, df: Union[HDataFrame, VDataFrame, MixDataFrame]
    ) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        """
        Discretize the data.

        Args:
            df: the X to discretize.

        Returns:
            the transformed X in federated dataframe.
        """

        assert hasattr(self, '_discretizer'), 'Discretizer has not been fit yet.'
        self._check_dataframe(df)
        assert len(df.columns) == len(self._discretizer.n_bins_), (
            f'X has {len(df.columns)} features but KBinsDiscretizer'
            f'is expecting {len(self._discretizer.n_bins_)} features as input.'
        )
        if isinstance(df, (HDataFrame, VDataFrame)):
            return self._transform(df, self._discretizer)
        else:
            # MixDataFrame
            new_parts = []
            if df.partition_way == PartitionWay.HORIZONTAL:
                for part in df.partitions:
                    new_parts.append(self._transform(part, self._discretizer))
            else:
                start_idx = 0
                end_idx = 0
                for part in df.partitions:
                    end_idx += len(part.columns)
                    est_part = SkKBinsDiscretizer(
                        n_bins=self._n_bins,
                        encode=self._encode,
                        strategy=self._strategy,
                    )
                    est_part.bin_edges_ = self._discretizer.bin_edges_[
                        start_idx:end_idx
                    ]
                    est_part.n_bins_ = self._discretizer.n_bins_[start_idx:end_idx]
                    new_parts.append(self._transform(part, est_part))
                    start_idx = end_idx
            return MixDataFrame(partitions=new_parts)

    def fit_transform(
        self,
        df: Union[HDataFrame, VDataFrame, MixDataFrame],
        aggregator: Aggregator = None,
        comparator: Comparator = None,
        compress_thres: int = 10000,
        error: float = 1e04,
        max_iter: int = 200,
    ):
        """Fit the estimator with X and then transform.
        Just a convience combine of fit and transform methods.
        """
        self.fit(
            df,
            aggregator=aggregator,
            comparator=comparator,
            compress_thres=compress_thres,
            error=error,
            max_iter=max_iter,
        )
        return self.transform(df)

    def get_params(self) -> Dict[str, Any]:
        assert hasattr(self, '_discretizer'), 'Discretizer has not been fit yet.'

        return {
            'n_bins': self._n_bins,
            'strategy': self._strategy,
            'encode': self._encode,
            'columns': self._columns,
            'discretizer': {
                'n_bins': self._discretizer.n_bins_,
                'bin_edges': self._discretizer.bin_edges_,
            },
        }
