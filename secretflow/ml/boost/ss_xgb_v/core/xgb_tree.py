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


class XgbTree:
    def __init__(self) -> None:
        self.split_features = list()
        self.split_values = list()

    def insert_split_node(self, feature: int, value: float) -> None:
        assert isinstance(feature, int), f"feature {feature}"
        assert isinstance(value, float), f"value {value}"
        self.split_features.append(feature)
        self.split_values.append(value)
