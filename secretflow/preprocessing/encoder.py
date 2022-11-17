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

from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from sklearn.utils.validation import column_or_1d

from secretflow.data.base import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.mix import MixDataFrame
from secretflow.data.vertical import VDataFrame
from secretflow.device import reveal


def _check_dataframe(df):
    assert isinstance(
        df, (HDataFrame, VDataFrame, MixDataFrame)
    ), f'Accepts HDataFrame/VDataFrame/MixDataFrame only but got {type(df)}'


class LabelEncoder:
    """Encode target labels with value between 0 and n_classes-1.

    Just same as :py:class:`sklearn.preprocessing.LabelEncoder`
    where the input/ouput is federated dataframe.

    Attributes:
        _encoder: the sklearn LabelEncoder instance.

    Examples:
        >>> from secretflow.preprocessing import LabelEncoder
        >>> le = LabelEncoder()
        >>> le.fit(df)
        >>> le.transform(df)
    """

    def _fit(self, df: Union[HDataFrame, VDataFrame]) -> np.ndarray:
        def _df_fit(df: pd.DataFrame):
            encoder = SkLabelEncoder()
            encoder.fit(column_or_1d(df))
            return encoder.classes_

        classes = [
            reveal(device(_df_fit)(part.data)) for device, part in df.partitions.items()
        ]
        return np.concatenate(classes)

    def fit(self, df: Union[HDataFrame, VDataFrame, MixDataFrame]):
        """Fit label encoder."""
        _check_dataframe(df)
        assert (
            len(df.columns) == 1
        ), 'DataFrame to encode should have one and only one column.'
        if isinstance(df, (HDataFrame, VDataFrame)):
            classes = self._fit(df)
        else:
            classes = np.concatenate([self._fit(part) for part in df.partitions])
        self._encoder = SkLabelEncoder()
        self._encoder.fit(classes)

    def _transform(
        self, df: Union[HDataFrame, VDataFrame]
    ) -> Union[HDataFrame, VDataFrame]:
        def _df_transform(encoder: SkLabelEncoder, df: pd.DataFrame):
            return pd.DataFrame(
                data=encoder.transform(column_or_1d(df))[np.newaxis].T,
                columns=df.columns,
            )

        transformed_parts = {}
        for device, part in df.partitions.items():
            transformed_parts[device] = Partition(
                device(_df_transform)(self._encoder, part.data)
            )

        new_df = df.copy()
        new_df.partitions = transformed_parts
        return new_df

    def transform(
        self, df: Union[HDataFrame, VDataFrame, MixDataFrame]
    ) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        """Transform labels to normalized encoding."""
        assert hasattr(self, '_encoder'), 'Encoder has not been fit yet.'
        _check_dataframe(df)
        assert (
            len(df.columns) == 1
        ), 'DataFrame to encode should have one and only one column.'

        if isinstance(df, (HDataFrame, VDataFrame)):
            return self._transform(df)
        else:
            return MixDataFrame(
                partitions=[self._transform(part) for part in df.partitions]
            )

    def fit_transform(
        self, df: Union[HDataFrame, VDataFrame]
    ) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        """Fit label encoder and return encoded labels."""
        self.fit(df)
        return self.transform(df)


class OneHotEncoder:
    """Encode categorical features as a one-hot numeric array.

    Just same as :py:class:`sklearn.preprocessing.OneHotEncoder`
    where the input/ouput is federated dataframe.

    Note: min_frequency and max_categories are calculated by partition,
        so they are only available for vertical scenarios currently.

    Args:
        min_frequency: int or float, default=None
            Specifies the minimum frequency below which a category will be
            considered infrequent.

            - If `int`, categories with a smaller cardinality will be considered
            infrequent.

            - If `float`, categories with a smaller cardinality than
            `min_frequency * n_samples`  will be considered infrequent.

        max_categories: int, default=None
            Specifies an upper limit to the number of output features for each input
            feature when considering infrequent categories. If there are infrequent
            categories, `max_categories` includes the category representing the
            infrequent categories along with the frequent categories. If `None`,
            there is no limit to the number of output features.

    Attributes:
        _encoder: the sklearn OneHotEncoder instance.

    Examples:
        >>> from secretflow.preprocessing import OneHotEncoder
        >>> enc = OneHotEncoder()
        >>> enc.fit(df)
        >>> enc.transform(df)
    """

    def __init__(self, min_frequency=None, max_categories=None):
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    @staticmethod
    def _fill_categories(categories: Dict[str, np.array]):
        """Fill the categories with the max length."""
        max_len = max([len(c) for c in categories.values()])
        for feature in categories.keys():
            fill_num = max_len - len(categories[feature])
            categories[feature] = np.append(
                categories[feature], [categories[feature][0] for i in range(fill_num)]
            )

    def _fit(self, df: Union[HDataFrame, VDataFrame]) -> Dict[str, np.array]:
        categories = []
        feature_names_in = []

        def _df_fit(df: pd.DataFrame):
            encoder = SkOneHotEncoder(min_frequency=self.min_frequency, max_categories=self.max_categories)
            encoder.fit(df)
            return encoder

        # reuse these encoders when min_frequency or max_categories are set
        self._encoders = {device: device(_df_fit)(part.data) for device, part in df.partitions.items()}
        if self.min_frequency or self.max_categories:
            return None

        encoders = reveal(list(self._encoders.values()))
        for encoder in encoders:
            if isinstance(df, HDataFrame):
                if len(feature_names_in) == 0:
                    feature_names_in = encoder.feature_names_in_
                if not categories:
                    categories = encoder.categories_
                else:
                    for i, category in enumerate(encoder.categories_):
                        categories[i] = np.concatenate([categories[i], category])
            elif isinstance(df, VDataFrame):
                categories.extend(encoder.categories_)
                feature_names_in.extend(encoder.feature_names_in_)
        assert len(feature_names_in) == len(
            categories
        ), f'Feature names length not equals to categories: {len(feature_names_in)} vs {len(categories)}'
        return {
            feature: category for feature, category in zip(feature_names_in, categories)
        }

    def fit(
        self, df: Union[HDataFrame, VDataFrame, MixDataFrame]
    ) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        """Fit this encoder with X."""
        _check_dataframe(df)
        if self.min_frequency or self.max_categories:
            assert (
                isinstance(df, VDataFrame)
            ), f'Args min_frequency/max_categories are only supported in VDataFrame'

        if isinstance(df, (HDataFrame, VDataFrame)):
            categories = self._fit(df)
        else:
            categories_list = [self._fit(part) for part in df.partitions]
            categories = categories_list[0]
            for cat in categories_list[1:]:
                for feature, category in cat.items():
                    if feature in categories:
                        categories[feature] = np.append(categories[feature], category)
                    else:
                        categories[feature] = category

        self._fitted = True
        if categories is None:
            return

        self._fill_categories(categories)

        self._encoder = SkOneHotEncoder()
        self._encoder.fit(pd.DataFrame(categories))

    def _transform(
        self, df: Union[HDataFrame, VDataFrame]
    ) -> Union[HDataFrame, VDataFrame]:
        transformed_parts = {}

        def _encode(encoder: SkOneHotEncoder, df: pd.DataFrame):
            return pd.DataFrame(
                encoder.transform(df).toarray(), columns=encoder.get_feature_names_out()
            )

        for device, part in df.partitions.items():
            if self.min_frequency or self.max_categories:
                # reuse encoder when min_frequency or max_categories are set
                encoder = self._encoders[device]
            else:
                encoder = SkOneHotEncoder()
                mask = np.in1d(
                    self._encoder.feature_names_in_, part.dtypes.index, assume_unique=True
                )
                categories = {
                    self._encoder.feature_names_in_[i]: self._encoder.categories_[i]
                    for i, exist in enumerate(mask)
                    if exist
                }
                self._fill_categories(categories)
                encoder.fit(pd.DataFrame(categories))
            transformed_parts[device] = Partition(device(_encode)(encoder, part.data))
        new_df = df.copy()
        new_df.partitions = transformed_parts
        return new_df

    def transform(
        self, df: Union[HDataFrame, VDataFrame, MixDataFrame]
    ) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        """Transform X using one-hot encoding."""
        assert hasattr(self, '_fitted'), 'Encoder has not been fit yet.'
        _check_dataframe(df)
        if isinstance(df, (HDataFrame, VDataFrame)):
            return self._transform(df)
        else:
            return MixDataFrame(
                partitions=[self._transform(part) for part in df.partitions]
            )

    def fit_transform(
        self, df: Union[HDataFrame, VDataFrame, MixDataFrame]
    ) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        """Fit this OneHotEncoder with X, then transform X."""
        self.fit(df)
        return self.transform(df)
