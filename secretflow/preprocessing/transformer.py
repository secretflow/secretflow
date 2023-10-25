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

from functools import partial
from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer as SkFunctionTransformer

from secretflow.data.horizontal import HDataFrame
from secretflow.data.mix import MixDataFrame
from secretflow.data.vertical import VDataFrame
from secretflow.preprocessing.base import _PreprocessBase


def _check_dataframe(df):
    assert isinstance(
        df, (HDataFrame, VDataFrame, MixDataFrame)
    ), f'Accepts HDataFrame/VDataFrame/MixDataFrame only but got {type(df)}'


class _FunctionTransformer(_PreprocessBase):
    """Constructs a transformer from an arbitrary callable.

    Just same as :py:class:`sklearn.preprocessing.FunctionTransformer`
    where the input/ouput is federated dataframe.

    Args:
        func: callable, default=None
            The callable to use for the transformation.
            If func is None, then func will be the identity function.
            Lambda is not supported here.

        kw_args: dict, default=None
            Dictionary of additional keyword arguments to pass to func.

    Attributes:
        _transformer: the sklearn FunctionTransformer instance.

    Examples:
        >>> from secretflow.preprocessing import _FunctionTransformer
        >>> ft = _FunctionTransformer(np.log1p)
        >>> ft.fit(df)
        >>> ft.transform(df)
    """

    def __init__(self, func: Callable, kw_args: Dict = None):
        self._transformer = SkFunctionTransformer(func=func, kw_args=kw_args)

    def _fit(self, df: Union[HDataFrame, VDataFrame]) -> np.ndarray:
        def _df_fit(_df):
            self._transformer.fit(_df)

        for device, part in df.partitions.items():
            device(_df_fit)(part.data)

    def fit(self, df: Union[HDataFrame, VDataFrame, MixDataFrame]):
        """Fit label encoder."""
        _check_dataframe(df)

        self._columns = df.columns
        if isinstance(df, (HDataFrame, VDataFrame)):
            self._fit(df)
        else:
            for part in df.partitions:
                self._fit(part)

    def _transform(
        self, df: Union[HDataFrame, VDataFrame]
    ) -> Union[HDataFrame, VDataFrame]:
        def _df_transform(_df: pd.DataFrame, transformer: SkFunctionTransformer):
            transformed_df = transformer.transform(_df)
            if isinstance(_df, pd.DataFrame):
                return pd.DataFrame(
                    data=transformed_df,
                    columns=_df.columns,
                )
            else:
                try:
                    import polars as pl

                    if isinstance(_df, pl.DataFrame):
                        if isinstance(transformed_df, pl.DataFrame):
                            return transformed_df
                        else:
                            return pl.DataFrame(
                                data=transformer.transform(_df),
                                schema={
                                    _df.columns[i]: _df.dtypes[i]
                                    for i in range(len(_df.columns))
                                },
                            )
                except ImportError:
                    pass
                raise RuntimeError(f"Unknown df type {type(_df)}")

        transformed_parts = {}
        for device, part in df.partitions.items():
            transformed_parts[device] = part.apply_func(
                _df_transform, transformer=self._transformer
            )
        new_df = df.copy()
        new_df.partitions = transformed_parts
        return new_df

    def transform(
        self, df: Union[HDataFrame, VDataFrame, MixDataFrame]
    ) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        """Transform labels with function."""
        _check_dataframe(df)

        if isinstance(df, (HDataFrame, VDataFrame)):
            return self._transform(df)
        else:
            return MixDataFrame(
                partitions=[self._transform(part) for part in df.partitions]
            )

    def fit_transform(
        self, df: Union[HDataFrame, VDataFrame]
    ) -> Union[HDataFrame, VDataFrame, MixDataFrame]:
        """Fit function transformer and return transformed DataFrame."""
        self.fit(df)
        return self.transform(df)

    def get_params(self) -> Dict[str, Any]:
        return {
            'columns': self._columns,
            'func': self._transformer.func,
            'kw_args': self._transformer.kw_args,
        }


class LogroundTransformer(_FunctionTransformer):
    """Constructs a transformer for calculating round(log2(x + bias)) of (partition of) dataframe.

    Args:
        decimals: Number of decimal places to round each column to. Defaults to 6.
        bias: Add bias to value before log2. Defaults to 0.5.

    """

    def __init__(self, decimals: int = 6, bias: float = 0.5):
        def _loground(
            x: Union[pd.DataFrame, "pl.DataFrame"],
            _decimals: int = 6,
            _bias: float = 0.5,
        ) -> pd.DataFrame:
            if isinstance(x, pd.DataFrame):
                return x.add(_bias).apply(np.log2).round(decimals=_decimals)
            else:
                try:
                    import polars as pl

                    if isinstance(x, pl.DataFrame):
                        x = x.with_columns(
                            pl.col(pl.NUMERIC_DTYPES)
                            .add(_bias)
                            .apply(np.log2)
                            .round(_decimals)
                        )
                        return x
                except ImportError:
                    pass
                raise RuntimeError(f"Unknown df type {type(x)}")

        self._decimals = decimals
        self._bias = bias
        super().__init__(partial(_loground, _decimals=decimals, _bias=bias))

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params['decimals'] = self._decimals
        params['bias'] = self._bias
        return params
