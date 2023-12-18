# Copyright 2023 Ant Group Co., Ltd.
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

from ray.tune.search.ax import AxSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.dragonfly import DragonflySearch
from ray.tune.search.flaml import CFO, BlendSearch
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.sigopt import SigOptSearch
from ray.tune.search.skopt import SkOptSearch
from ray.tune.search.zoopt import ZOOptSearch

OptunaSearch = OptunaSearch
AxSearch = AxSearch
BayesOptSearch = BayesOptSearch
TuneBOHB = TuneBOHB
BlendSearch = BlendSearch
CFO = CFO
DragonflySearch = DragonflySearch
HEBOSearch = HEBOSearch
HyperOptSearch = HyperOptSearch
NevergradSearch = NevergradSearch
SigOptSearch = SigOptSearch
SkOptSearch = SkOptSearch
ZOOptSearch = ZOOptSearch
