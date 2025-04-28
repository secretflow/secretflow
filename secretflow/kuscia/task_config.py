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

import functools
import logging
from dataclasses import dataclass
from typing import Dict, List

from google.protobuf import json_format
from kuscia.proto.api.v1alpha1.kusciatask.kuscia_task_pb2 import (
    AllocatedPorts,
    ClusterDefine,
)

from secretflow.spec.extend.cluster_pb2 import SFClusterDesc
from secretflow.spec.v1.data_pb2 import StorageConfig
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


@dataclass
class ColumnAttr:
    col_name: str = None
    col_type: str = None


@dataclass
class TableAttr:
    table_id: str = None
    column_attrs: List[ColumnAttr] = None


@dataclass
class KusciaTaskConfig:
    task_id: str
    task_cluster_def: ClusterDefine
    task_allocated_ports: AllocatedPorts
    task_progress_url: str = ""
    sf_node_eval_param: NodeEvalParam = None
    sf_cluster_desc: SFClusterDesc = None
    sf_storage_config: Dict[str, StorageConfig] = None
    sf_input_ids: List[str] = None
    sf_input_partitions_spec: List[str] = None
    sf_output_ids: List[str] = None
    sf_output_uris: List[str] = None
    sf_output_partitions_spec: List[str] = None
    sf_datasource_config: Dict[str, Dict] = None
    table_attrs: List[TableAttr] = None

    @functools.cached_property
    def party_name(self):
        party_id = self.task_cluster_def.self_party_idx
        return self.task_cluster_def.parties[party_id].name

    # NOTE(junfeng): the format of kuscia task is subject to modify.
    @classmethod
    def from_json(cls, req: Dict):
        logging.info(f"KusciaTaskConfig.from_json: {req}")
        task_id = req["task_id"]
        task_cluster_def = ClusterDefine()
        json_format.Parse(req["task_cluster_def"], task_cluster_def)
        task_allocated_ports = AllocatedPorts()
        json_format.Parse(req["allocated_ports"], task_allocated_ports)
        task_progress_url = req.get("task_progress_url", "")

        if "task_input_config" in req:
            sf_node_eval_param = NodeEvalParam()
            json_format.ParseDict(
                req["task_input_config"]["sf_node_eval_param"], sf_node_eval_param
            )
            sf_cluster_desc = SFClusterDesc()
            json_format.ParseDict(
                req["task_input_config"]["sf_cluster_desc"], sf_cluster_desc
            )

            sf_storage_config = None
            if "sf_storage_config" in req["task_input_config"]:
                sf_storage_config = {}
                for storage_party, config in req["task_input_config"][
                    "sf_storage_config"
                ].items():
                    storage_config_pb = StorageConfig()
                    json_format.ParseDict(config, storage_config_pb)
                    sf_storage_config[storage_party] = storage_config_pb

            sf_input_ids = (
                req["task_input_config"]["sf_input_ids"]
                if "sf_input_ids" in req["task_input_config"]
                else None
            )
            sf_input_partitions_spec = (
                req["task_input_config"]["sf_input_partitions_spec"]
                if "sf_input_partitions_spec" in req["task_input_config"]
                else None
            )
            sf_output_ids = (
                req["task_input_config"]["sf_output_ids"]
                if "sf_output_ids" in req["task_input_config"]
                else None
            )
            sf_output_uris = (
                req["task_input_config"]["sf_output_uris"]
                if "sf_output_uris" in req["task_input_config"]
                else None
            )
            sf_output_partitions_spec = (
                req["task_input_config"]["sf_output_partitions_spec"]
                if "sf_output_partitions_spec" in req["task_input_config"]
                else None
            )

            sf_datasource_config = None
            if "sf_datasource_config" in req["task_input_config"]:
                sf_datasource_config = {}
                for party, config in req["task_input_config"][
                    "sf_datasource_config"
                ].items():
                    sf_datasource_config[party] = config

            table_attrs_def = []
            if "table_attrs" in req["task_input_config"]:
                for table_attr in req["task_input_config"]["table_attrs"]:
                    table_attrs = TableAttr(
                        table_id=table_attr["table_id"],
                        column_attrs=[
                            ColumnAttr(
                                col_name=col["col_name"], col_type=col["col_type"]
                            )
                            for col in table_attr["column_attrs"]
                        ],
                    )
                    table_attrs_def.append(table_attrs)

            return cls(
                task_id,
                task_cluster_def,
                task_allocated_ports,
                task_progress_url,
                sf_node_eval_param,
                sf_cluster_desc,
                sf_storage_config,
                sf_input_ids,
                sf_input_partitions_spec,
                sf_output_ids,
                sf_output_uris,
                sf_output_partitions_spec,
                sf_datasource_config,
                table_attrs=table_attrs_def,
            )
        else:
            return cls(
                task_id,
                task_cluster_def,
                task_allocated_ports,
                task_progress_url,
            )

    @classmethod
    def from_file(cls, task_config_path: str):
        with open(task_config_path) as f:
            import json

            configs = json.load(f)
            configs["task_input_config"] = json.loads(configs["task_input_config"])
            return cls.from_json(configs)
