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
from dataclasses import dataclass
from typing import Tuple, Union

from secretflow.data import FedNdarray
from secretflow.device import PYUObject
from secretflow.ml.boost.core.data_preprocess import (
    validate,
    validate_sample_weight,
    validate_tweedie_label,
)
from secretflow.ml.boost.sgb_v.core.params import default_params

from ....core.params import RegType
from ..component import Component, print_params


@dataclass
class DataPreprocessParams:
    """
    'objective': Specify the learning objective.
        default: 'logistic'
        range: ['linear', 'logistic', 'tweedie']
    """

    objective: RegType = default_params.objective


class DataPreprocessor(Component):
    def __init__(self) -> None:
        super().__init__()
        self.params = DataPreprocessParams()

    def show_params(self):
        print_params(self.params)

    def set_params(self, params: dict):
        obj = params.get('objective', 'logistic')
        obj = RegType(obj)
        self.params.objective = obj

    def get_params(self, params: dict):
        params['objective'] = self.params.objective

    def set_devices(self, _):
        return

    def set_actors(self, _):
        return

    def del_actors(self):
        return

    def validate(
        self, dataset, label, sample_weight=None
    ) -> Tuple[
        FedNdarray, Tuple[int, int], PYUObject, Tuple[int, int], Union[None, PYUObject]
    ]:
        x, x_shape, y, y_shape = validate(dataset, label)
        # tweedie regression only support non negative labels
        if self.params.objective == RegType.Tweedie:
            validate_tweedie_label(y)
        w = validate_sample_weight(sample_weight, y_shape=y_shape)
        return x, x_shape, y, y_shape, w
