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
from typing import Dict, List, Union

import spu
import yaml
from google.protobuf import json_format

from secretflow.protos.component.node_def_pb2 import NodeDef
from secretflow.protos.kuscia.kuscia_task_pb2 import AllocatedPorts, ClusterDefine


@dataclass
class TaskConfig:
    task_id: str = None
    party_id: int = None
    party_name: str = None

    ray_node_ip_address: str = None
    ray_gcs_port: int = None
    ray_node_manager_port: int = None
    ray_object_manager_port: int = None
    ray_client_server_port: int = None
    ray_worker_ports: List[int] = None
    spu_port: int = None
    fed_port: int = None

    cluster_config: Dict[str, Union[str, Dict[str, str]]] = None

    spu_cluster_config: Dict[
        str,
        Union[
            List[str],
            Dict[str, Union[List[Dict[str, str]], Dict[str, Union[int, bool]]]],
        ],
    ] = None

    comp_node: NodeDef = None

    def parse_from_file(self, task_config_path: str):
        with open(task_config_path) as f:
            cluster_define = ClusterDefine()
            allocated_port = AllocatedPorts()
            self.comp_node = NodeDef()

            configs = yaml.safe_load(f)
            self.task_id = configs['task_id']
            json_format.Parse(configs['task_input_config'], self.comp_node)
            json_format.Parse(configs['task_input_cluster_def'], cluster_define)
            json_format.Parse(configs['allocated_ports'], allocated_port)

            self.party_id = cluster_define.self_party_idx
            self.party_name = cluster_define.parties[self.party_id].name
            self.ray_worker_ports = []
            self.cluster_config = {}

            self.spu_cluster_config = {}
            self.spu_cluster_config["pyu"] = []
            self.spu_cluster_config["spu"] = {"nodes": []}

            for port in allocated_port.ports:
                if port.name.startswith('ray-worker'):
                    self.ray_worker_ports.append(port.port)
                elif port.name == 'spu':
                    self.spu_port = port.port
                elif port.name == 'node-manager':
                    self.ray_node_manager_port = port.port
                elif port.name == 'object-manager':
                    self.ray_object_manager_port = port.port
                elif port.name == 'client-server':
                    self.ray_client_server_port = port.port
                elif port.name == 'fed':
                    self.fed_port = port.port

            self.cluster_config['parties'] = {}
            for party in cluster_define.parties:
                self.spu_cluster_config["pyu"].append(party.name)
                if party.name != self.party_name:
                    for service in party.services:
                        if service.port_name == 'fed':
                            if len(service.endpoints[0].split(':')) < 2:
                                service.endpoints[0] += ':80'
                            self.cluster_config['parties'][party.name] = {
                                'address': service.endpoints[0]
                            }
                        elif service.port_name == 'spu':
                            if len(service.endpoints[0].split(':')) < 2:
                                service.endpoints[0] += ':80'
                            self.spu_cluster_config["spu"]["nodes"].append(
                                {
                                    'party': party.name,
                                    'id': f'{party.name}:0',
                                    # add "http://" to force brpc to set the correct Host
                                    'address': f'http://{service.endpoints[0]}',
                                }
                            )
                else:
                    for service in cluster_define.parties[self.party_id].services:
                        if service.port_name == 'global':
                            segs = service.endpoints[0].split(':')
                            self.ray_node_ip_address = segs[0]
                            if len(segs) == 2:
                                self.ray_gcs_port = int(segs[1])
                            else:
                                self.ray_gcs_port = 80

            self.cluster_config['parties'][self.party_name] = {
                'address': f'0.0.0.0:{self.fed_port}'
            }
            self.cluster_config['self_party'] = self.party_name

            self.spu_cluster_config['spu']['nodes'].append(
                {
                    'party': self.party_name,
                    'id': f'{self.party_name}:0',
                    'address': f'0.0.0.0:{self.spu_port}',
                }
            )
            self.spu_cluster_config['spu']['nodes'].sort(key=lambda x: x['party'])

            if len(cluster_define.parties) == 2:
                self.spu_cluster_config['spu']['runtime_config'] = {
                    'protocol': spu.spu_pb2.SEMI2K,
                    'field': spu.spu_pb2.FM128,
                }
            elif len(cluster_define.parties) == 3:
                self.spu_cluster_config['spu']['runtime_config'] = {
                    'protocol': spu.spu_pb2.ABY3,
                    'field': spu.spu_pb2.FM64,
                }

        return self
