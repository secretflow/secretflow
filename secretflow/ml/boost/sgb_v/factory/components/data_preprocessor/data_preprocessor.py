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

from typing import Tuple

from secretflow.data import FedNdarray
from secretflow.device import PYUObject
from secretflow.ml.boost.core.data_preprocess import validate

from ..component import Component


class DataPreprocessor(Component):
    def __init__(self) -> None:
        super().__init__()

    def show_params(self):
        return

    def set_params(self, _):
        return

    def get_params(self, _):
        return

    def set_devices(self, _):
        return

    def set_actors(self, _):
        return

    def del_actors(self):
        return

    def validate(
        self, dataset, label
    ) -> Tuple[FedNdarray, Tuple[int, int], PYUObject, Tuple[int, int]]:
        return validate(dataset, label)
