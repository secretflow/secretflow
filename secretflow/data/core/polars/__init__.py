# Copyright 2023 Ant Group Co., Ltd.
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

import sys

PlPartDataFrame = None

try:
    from .dataframe import PlPartDataFrame
    from .util import infer_pl_dtype, read_polars_csv
except ImportError as exc:
    exc_tb = sys.exc_info()[2]
    msg = (
        "Secretflow's partitions bindings (polars) is not installed. "
        "Please run `pip install polars` by yourself."
    )
    raise type(exc)(msg).with_traceback(exc_tb) from None

__all__ = ["PlPartDataFrame", "infer_pl_dtype", "read_polars_csv"]
