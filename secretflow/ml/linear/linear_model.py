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


from enum import Enum, unique
from dataclasses import dataclass
from typing import Union, List
from secretflow.utils.sigmoid import SigType
from secretflow.device import SPUObject, PYUObject


@unique
class RegType(Enum):
    Linear = 'linear'
    Logistic = 'logistic'


@dataclass
class LinearModel:
    """
    Unified linear regression model.

    Attributes:

        weights : {SPUObject, List[PYUObject]}
            for mpc lr, use SPUObject save all weights; for fl lr, use list of PYUObject.
        reg_type : RegType
            linear regression or logistic regression model.
        sig_type : SigType
            which sigmoid approximation should use, only use in mpc lr.
    """

    weights: Union[SPUObject, List[PYUObject]]
    reg_type: RegType
    sig_type: SigType
