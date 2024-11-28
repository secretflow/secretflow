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


class XgbConfig:
    def __init__(self, config: dict):
        self._config = config

    @property
    def use_completely_sgb(self) -> bool:
        return self._config.get('use_completely_sgb', False)

    @use_completely_sgb.setter
    def use_completely_sgb(self, value: bool):
        self._config['use_completely_sgb'] = value

    # called by parties that do not hold label colum
    @property
    def support_completely_sgb(self) -> bool:
        return self._config.get('support_completely_sgb', False)

    # called by parties that do not hold label colum
    @property
    def support_row_sample_by_tree(self) -> bool:
        return self._config.get('support_row_sample_by_tree', False)

    # called by parties that do not hold label colum
    @property
    def support_col_sample_by_tree(self) -> bool:
        return self._config.get('support_col_sample_by_tree', False)

    @property
    def num_round(self) -> int:
        return self._config['num_round']

    @num_round.setter
    def num_round(self, value: int):
        self._config['num_round'] = value

    @property
    def max_depth(self) -> int:
        return self._config['max_depth']

    @max_depth.setter
    def max_depth(self, value: int):
        self._config['max_depth'] = value

    @property
    def row_sample_by_tree(self) -> float:
        return self._config['row_sample_by_tree']

    @row_sample_by_tree.setter
    def row_sample_by_tree(self, value: float):
        self._config['row_sample_by_tree'] = value

    @property
    def col_sample_by_tree(self) -> float:
        return self._config['col_sample_by_tree']

    @col_sample_by_tree.setter
    def col_sample_by_tree(self, value: float):
        self._config['col_sample_by_tree'] = value

    @property
    def bucket_eps(self) -> float:
        return self._config['bucket_eps']

    @bucket_eps.setter
    def bucket_eps(self, value: float):
        self._config['bucket_eps'] = value

    # called by the party holding label column
    @property
    def objective(self) -> str:
        return self._config['objective']

    # called by the party holding label column
    @property
    def reg_lambda(self) -> float:
        return self._config['reg_lambda']

    # called by the party holding label column
    @property
    def gamma(self) -> float:
        return self._config['gamma']
