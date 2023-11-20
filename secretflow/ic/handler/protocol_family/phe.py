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


from secretflow.ic.proto.handshake.protocol_family import phe_pb2


class PheConfig:
    def __init__(self, config: dict):
        self._config = config

    @property
    def phe_algo(self) -> phe_pb2.PheAlgo:
        schema = self._config['he_parameters']['schema']
        assert schema.startswith('ic-')
        schema = schema[3:]
        algo_name = 'PHE_ALGO_' + schema.upper()
        return phe_pb2.PheAlgo.Value(algo_name)

    @phe_algo.setter
    def phe_algo(self, value):
        name = phe_pb2.PheAlgo.Name(value)[9:]  # remove prefix
        self._config['he_parameters']['schema'] = 'ic-' + name.lower()

    @property
    def key_size(self) -> int:
        return self._config['he_parameters']['key_pair']['generate']['bit_size']

    @property
    def config(self) -> dict:
        return self._config
