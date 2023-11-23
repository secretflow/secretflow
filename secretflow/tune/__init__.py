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

import sys
import ray

try:
    import ray.tune

except ImportError as exc:
    exc_tb = sys.exc_info()[2]
    msg = (
        "Secretflow tune requires ray[tune] which is not installed."
        f"Please run `pip install ray[tune]=={ray.__version__}` by yourself."
    )
    raise type(exc)(msg).with_traceback(exc_tb) from None

from . import train
from .result_grid import ResultGrid
from .search import grid_search
from .trainable import with_parameters, with_resources
from .tune_config import TuneConfig
from .tuner import Tuner
