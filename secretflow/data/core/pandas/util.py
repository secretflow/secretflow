# Copyright 2024 Ant Group Co., Ltd.
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

import pandas as pd


def read_pandas_csv(filepath, *args, **kwargs):
    header_row = 0
    if 'header' in kwargs and kwargs['header'] is None:
        header_row = -1

    skiprows = kwargs.pop('skip_rows_after_header', None)
    if skiprows is not None:
        assert isinstance(skiprows, int)
        skiprows += header_row + 1

        def skip_rows(r):
            return r > header_row and r < skiprows

        kwargs['skiprows'] = skip_rows
        try:
            df = pd.read_csv(open(filepath), *args, **kwargs)
        except pd.errors.EmptyDataError:
            # skip ending with empty df, not exception
            df = pd.DataFrame()
    else:
        df = pd.read_csv(open(filepath), *args, **kwargs)

    if 'usecols' in kwargs and kwargs['usecols'] is not None:
        return df[kwargs['usecols']]
    else:
        return df
