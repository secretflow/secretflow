# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import pandas as pd


def read_csv_wrapper(
    filepath: str, auto_gen_header_prefix: str = "", read_backend="pandas", **kwargs
) -> Union[pd.DataFrame, "pl.DataFrame"]:
    """A wrapper of pandas read_csv and supports oss file.

    Args:
        filepath: the file path.
        auto_gen_header_prefix: If set, the format of generated headers would be {gen_header_prefix}_{col_idx}.
        read_backend: reading backend.
        kwargs: all other arguments are same with :py:meth:`pandas.DataFrame.read_csv`.

    Returns:
        a DataFrame.
    """

    def _read_csv(_filepath, _backend, *_args, **_kwargs):
        if _backend == "pandas":
            return pd.read_csv(open(_filepath), *_args, **_kwargs)
        elif _backend == "polars":
            from secretflow.data.core.polars.util import read_polars_csv

            return read_polars_csv(_filepath, *_args, **_kwargs)
        else:
            raise RuntimeError(f"Unknown data backend {_backend}")

    if auto_gen_header_prefix:
        kwargs['header'] = None
        df = _read_csv(filepath, read_backend, **kwargs)
        df.columns = [
            "{}_{}".format(auto_gen_header_prefix, i) for i in range(df.shape[1])
        ]
        return df
    else:
        df = _read_csv(filepath, read_backend, **kwargs)
        return df
