# Copyright 2024 Ant Group Co., Ltd.
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


import abc
from dataclasses import dataclass

from secretflow_spec import Storage, VTableFormat, VTableSchema


@dataclass
class TableInfo:
    schema: VTableSchema
    line_count: int


class IConnector(abc.ABC):
    '''
    like flink, connector provide code for interfacing with various third-party systems
    '''

    @abc.abstractmethod
    def download_table(
        self,
        storage: Storage,
        data_dir: str,
        input_path: str,
        input_params: dict,
        output_uri: str,
        output_format: VTableFormat = VTableFormat.ORC,
    ) -> TableInfo:
        pass

    @abc.abstractmethod
    def upload_table(
        self,
        storage: Storage,
        data_dir: str,
        input_uri: str,
        input_format: VTableFormat,
        input_schema: VTableSchema,
        output_path: str,
        output_params: dict,
    ):
        pass
