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

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

from secretflow.data.horizontal import HDataFrame
from secretflow.data.mix import MixDataFrame
from secretflow.data.vertical import VDataFrame
from secretflow.preprocessing.base import _PreprocessBase


def _check_dataframe(df):
    assert isinstance(
        df, (HDataFrame, VDataFrame, MixDataFrame)
    ), f'Accepts HDataFrame/VDataFrame/MixDataFrame only but got {type(df)}'


class VOrdinalEncoder(_PreprocessBase):
    """Encode categorical features as ordinal numbers.
    Just same as :py:class:`sklearn.preprocessing.OrdinalEncoder`
    where the input/output is federated dataframe.
    Only suppoer vertical ordinal encoding for now.


    Attributes:
        _encoder: the sklearn OrdinalEncoder instance.
    Examples:
        >>> from secretflow.preprocessing import OrdinalEncoder
        >>> oe = OrdinalEncoder()
        >>> oe.fit(df)
        >>> oe.transform(df)
    """

    def fit(self, df: VDataFrame) -> np.ndarray:
        def _df_fit(_df: pd.DataFrame):
            encoder = SkOrdinalEncoder()
            encoder.fit(_df)
            return encoder

        self._encoders = {
            device: device(_df_fit)(part.data) for device, part in df.partitions.items()
        }

    def _transform(self, df: VDataFrame, inverse: bool = False) -> VDataFrame:
        def _df_transform(
            _df: pd.DataFrame,
            encoder: SkOrdinalEncoder,
            inverse: bool,
        ):
            if isinstance(_df, pd.DataFrame):
                if inverse:
                    return pd.DataFrame(
                        data=encoder.inverse_transform(_df),
                        columns=_df.columns,
                    )
                return pd.DataFrame(
                    data=encoder.transform(_df),
                    columns=_df.columns,
                )

        transformed_parts = {}

        for device, part in df.partitions.items():
            transformed_parts[device] = part.apply_func(
                _df_transform, encoder=self._encoders[device], inverse=inverse
            )
        new_df = df.copy()
        new_df.partitions = transformed_parts
        return new_df

    def inverse_transform(self, df: VDataFrame) -> VDataFrame:
        """Transform labels back to original encoding."""
        assert hasattr(self, '_encoders'), 'Encoder has not been fit yet.'
        _check_dataframe(df)
        if isinstance(df, VDataFrame):
            return self._transform(df, inverse=True)
        else:
            return MixDataFrame(
                partitions=[
                    self._transform(part, inverse=True) for part in df.partitions
                ]
            )

    def transform(self, df: VDataFrame) -> VDataFrame:
        """Transform labels to ordinal encoding."""
        assert hasattr(self, '_encoders'), 'Encoder has not been fit yet.'
        assert isinstance(df, VDataFrame), "Currently only supports VDataFrame"
        return self._transform(df)

    def fit_transform(self, df: VDataFrame) -> VDataFrame:
        """Fit the OrdinalEncoder and return encoded labels."""
        self.fit(df)
        return self.transform(df)

    def get_params(self) -> Any:
        """Not Supported for now"""
        return None
