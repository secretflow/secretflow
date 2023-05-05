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

import builtins
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import pandas as pd


def open(filepath: Union[str, Path], mode='rb'):
    """Open a oss/http/https file.

    Args:
        filepath: The file path, which can be an oss, or pathlib.Path object.
        mode: optional. open mode.

    Returns:
        the file object.
    """
    if not isinstance(filepath, str):
        return filepath

    o = urlparse(filepath)
    if o.scheme == 'oss':
        import secretflow.data.io.oss as oss

        return oss.open(filepath, mode)

    return builtins.open(filepath, mode)


def is_local_file(uri: str) -> bool:
    return uri and not urlparse(uri).scheme


def read_csv_wrapper(
    filepath: str, auto_gen_header_prefix: str = "", **kwargs
) -> pd.DataFrame:
    """A wrapper of pandas read_csv and supports oss file.

    Args:
        filepath: the file path.
        auto_gen_header_prefix: If set, the format of generated headers would be {gen_header_prefix}_{col_idx}.
        kwargs: all other arguments are same with :py:meth:`pandas.DataFrame.read_csv`.

    Returns:
        a pandas DataFrame.
    """
    if auto_gen_header_prefix:
        kwargs['header'] = None
        df = pd.read_csv(open(filepath), **kwargs)
        df.columns = [
            "{}_{}".format(auto_gen_header_prefix, i) for i in range(df.shape[1])
        ]
        return df
    else:
        return pd.read_csv(open(filepath), **kwargs)


def to_csv_wrapper(df: pd.DataFrame, filepath, **kwargs):
    """A wrapper of pandas to_csv and supports oss file.

    Args:
        filepath: the file path.
        kwargs: all other arguments are same with :py:meth:`pandas.DataFrame.read_csv`.

    Returns:
        a pandas DataFrame.
    """
    if is_local_file(filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(open(filepath, 'wb'), **kwargs)
