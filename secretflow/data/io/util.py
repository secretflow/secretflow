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
from io import BytesIO
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import pandas as pd
import requests

import secretflow.data.io.oss as oss


def open(filepath: Union[str, Path], mode='rb'):
    """打开文件，支持oss、http/https形式的文件。

    Args:
        filepath: 文件路径，可以是oss、http/https或者pathlib.Path对象。
        mode: 可选; 打开模式。

    Returns:
        文件对象。
    """
    if not isinstance(filepath, str):
        return filepath

    o = urlparse(filepath)
    if o.scheme == 'oss':
        return oss.open(filepath, mode)
    if o.scheme == 'http' or o.scheme == 'https':
        r = requests.get(filepath, stream=True)
        return BytesIO(r.raw.read())

    return builtins.open(filepath, mode)


def read_csv_wrapper(filepath, *arg, **kwargs):
    return pd.read_csv(open(filepath), *arg, **kwargs)
