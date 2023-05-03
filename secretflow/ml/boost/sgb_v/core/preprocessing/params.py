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

from dataclasses import dataclass
from enum import Enum, unique


@unique
class RegType(Enum):
    Linear = 'linear'
    Logistic = 'logistic'


@dataclass
class LabelHolderInfo:
    seed: int
    reg_lambda: float
    gamma: float
    learning_rate: float
    base_score: float
    sample_num: int
    subsample_rate: float
    obj_type: RegType


@dataclass
class SGBTrainParams:
    num_boost_round: int
    max_depth: int
    learning_rate: float
    objective: RegType
    reg_lambda: float
    gamma: float
    subsample: float
    colsample_by_tree: float
    base_score: float
    sketch_eps: float
    seed: int
    fixed_point_parameter: int
