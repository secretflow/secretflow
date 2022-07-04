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

from abc import ABC, abstractmethod
from typing import List

from secretflow.device import DeviceObject


class Aggregator(ABC):
    """The abstract aggregator."""

    @abstractmethod
    def sum(self, data: List[DeviceObject], axis=None):
        """Sum of array elements over a given axis."""
        pass

    @abstractmethod
    def average(self, data: List[DeviceObject], axis=None, weights=None):
        """Compute the weighted average along the specified axis."""
        pass
