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
from secretflow.device.device import reveal


def _check_dataframe(df):
    assert isinstance(df, (HDataFrame, VDataFrame, MixDataFrame)
                      ), f'Accepts HDataFrame/VDataFrame/MixDataFrame only but got {type(df)}'


class LabelEncoder:
    def _fit(self, df: Union[HDataFrame, VDataFrame]) -> np.ndarray:
        def _df_fit(df: pd.DataFrame):
            encoder = SkLabelEncoder()
            encoder.fit(column_or_1d(df))
            return encoder.classes_

        classes = [reveal(device(_df_fit)(part.data)) for device, part in df.partitions.items()]
        return np.concatenate(classes)

    def fit(self, df: Union[HDataFrame, VDataFrame, MixDataFrame]):
        _check_dataframe(df)
        assert len(df.columns) == 1, 'DataFrame to encode should have one and only one column.'
        if isinstance(df, (HDataFrame, VDataFrame)):
            classes = self._fit(df)
        else:
            classes = np.concatenate([self._fit(part) for part in df.partitions])
        self._encoder = SkLabelEncoder()
        self._encoder.fit(classes)

    def _transform(self, df: Union[HDataFrame, VDataFrame]) -> Union[HDataFrame, VDataFrame]:
        def _df_transform(encoder: SkLabelEncoder, df: pd.DataFrame):
            new_df = df.copy(deep=True)
            new_df.iloc[:, :] = encoder.transform(column_or_1d(df))[np.newaxis].T
            return new_df

        transformed_parts = {}
        for device, part in df.partitions.items():
            transformed_parts[device] = Partition(device(_df_transform)(self._encoder, part.data))

        new_df = df.copy()
        new_df.partitions = transformed_parts
        return new_df

    def transform(self, df: Union[HDataFrame, VDataFrame, MixDataFrame]) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        assert hasattr(self, '_encoder'), 'Encoder has not been fit yet.'
        _check_dataframe(df)
        assert len(df.columns) == 1, 'DataFrame to encode should have one and only one column.'

        if isinstance(df, (HDataFrame, VDataFrame)):
            return self._transform(df)
        else:
            return MixDataFrame(partitions=[self._transform(part) for part in df.partitions])

    def fit_transform(self, df: Union[HDataFrame, VDataFrame]) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        self.fit(df)
        return self.transform(df)


class OneHotEncoder:

    @staticmethod
    def _fill_categories(categories: Dict[str, np.array]):
        """使用第一个值填充categories，从而每个array的长度一致以便于后续使用。
        """
        max_len = max([len(c) for c in categories.values()])
        for feature in categories.keys():
            fill_num = max_len - len(categories[feature])
            categories[feature] = np.append(categories[feature], [categories[feature][0] for i in range(fill_num)])

    def _fit(self, df: Union[HDataFrame, VDataFrame]) -> Dict[str, np.array]:
        categories = []
        feature_names_in = []

        def _df_fit(df: pd.DataFrame):
            encoder = SkOneHotEncoder()
            encoder.fit(df)
            return encoder
        encoders = reveal([device(_df_fit)(part.data) for device, part in df.partitions.items()])
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
            categories), f'Feature names length not equals to categories: {len(feature_names_in)} vs {len(categories)}'
        return {feature: category for feature, category in zip(feature_names_in, categories)}

    def fit(self, df: Union[HDataFrame, VDataFrame, MixDataFrame]) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        _check_dataframe(df)
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
        self._fill_categories(categories)

        self._encoder = SkOneHotEncoder()
        self._encoder.fit(pd.DataFrame(categories))

    def _transform(self, df: Union[HDataFrame, VDataFrame]) -> Union[HDataFrame, VDataFrame]:
        transformed_parts = {}

        def _encode(encoder: SkOneHotEncoder, df: pd.DataFrame):
            return pd.DataFrame(encoder.transform(df).toarray(), columns=encoder.get_feature_names_out())
        for device, part in df.partitions.items():
            mask = np.in1d(self._encoder.feature_names_in_, part.dtypes.index, assume_unique=True)
            encoder = SkOneHotEncoder()
            categories = {self._encoder.feature_names_in_[
                i]: self._encoder.categories_[i] for i, exist in enumerate(mask) if exist}
            self._fill_categories(categories)
            encoder.fit(pd.DataFrame(categories))
            transformed_parts[device] = Partition(device(_encode)(encoder, part.data))
        new_df = df.copy()
        new_df.partitions = transformed_parts
        return new_df

    def transform(self, df: Union[HDataFrame, VDataFrame, MixDataFrame]) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        assert hasattr(self, '_encoder'), 'Encoder has not been fit yet.'
        _check_dataframe(df)
        if isinstance(df, (HDataFrame, VDataFrame)):
            return self._transform(df)
        else:
            return MixDataFrame(partitions=[self._transform(part) for part in df.partitions])

    def fit_transform(self, df: Union[HDataFrame, VDataFrame, MixDataFrame]) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        self.fit(df)
        return self.transform(df)
