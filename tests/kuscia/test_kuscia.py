# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from kuscia.proto.api.v1alpha1.common_pb2 import DataColumn
from kuscia.proto.api.v1alpha1.datamesh.domaindata_pb2 import DomainData
from kuscia.proto.api.v1alpha1.kusciatask.kuscia_task_pb2 import (
    AllocatedPorts,
    ClusterDefine,
    Party,
    Port,
    Service,
)
from secretflow_spec.v1.data_pb2 import DistData, IndividualTable, TableSchema

from secretflow.kuscia.entry import convert_domain_data_to_individual_table
from secretflow.kuscia.ray_config import RayConfig
from secretflow.kuscia.sf_config import get_sf_cluster_config
from secretflow.kuscia.task_config import KusciaTaskConfig
from secretflow.spec.extend.cluster_pb2 import SFClusterDesc


def test_load_configs():
    kuscia_request_json = {
        "task_id": "secretflow-task-20240705104523-single-psi",
        "task_cluster_def": '{"parties":[{"name":"alice","role":"","services":[{"portName":"spu","endpoints":["secretflow-task-20240705104523-single-psi-0-spu.alice.svc"]},{"portName":"fed","endpoints":["secretflow-task-20240705104523-single-psi-0-fed.alice.svc"]},{"portName":"global","endpoints":["secretflow-task-20240705104523-single-psi-0-global.alice.svc:25815"]},{"portName":"inference","endpoints":["alice_inference:1237"]}]},{"name":"bob","role":"","services":[{"portName":"spu","endpoints":["secretflow-task-20240705104523-single-psi-0-spu.bob.svc"]},{"portName":"fed","endpoints":["secretflow-task-20240705104523-single-psi-0-fed.bob.svc"]},{"portName":"global","endpoints":["secretflow-task-20240705104523-single-psi-0-global.bob.svc:30818"]},{"portName":"inference","endpoints":["bob_inference:1238"]}]}],"selfPartyIdx":0,"selfEndpointIdx":0}',
        "allocated_ports": '{"ports":[{"name":"inference","port":1237,"scope":"Cluster","protocol":"HTTP"},{"name":"spu","port":25813,"scope":"Cluster","protocol":"GRPC"},{"name":"fed","port":25814,"scope":"Cluster","protocol":"GRPC"},{"name":"global","port":25815,"scope":"Domain","protocol":"GRPC"},{"name":"node-manager","port":25810,"scope":"Local","protocol":"GRPC"},{"name":"object-manager","port":25811,"scope":"Local","protocol":"GRPC"},{"name":"client-server","port":25812,"scope":"Local","protocol":"GRPC"}]}',
    }

    kuscia_config = KusciaTaskConfig.from_json(kuscia_request_json)

    assert kuscia_config.task_id == "secretflow-task-20240705104523-single-psi"
    assert len(kuscia_config.task_cluster_def.parties) == 2
    assert len(kuscia_config.task_allocated_ports.ports) == 7

    ray_config = RayConfig.from_kuscia_task_config(kuscia_config)
    assert (
        ray_config.ray_node_ip_address
        == "secretflow-task-20240705104523-single-psi-0-global.alice.svc"
    )
    assert ray_config.ray_gcs_port == 25815
    assert (
        ray_config.ray_min_worker_port
        >= 10000 & ray_config.ray_min_worker_port
        <= 20000
    )
    assert (
        ray_config.ray_max_worker_port
        >= 10000 & ray_config.ray_max_worker_port
        <= 20000
    )
    assert ray_config.ray_max_worker_port - ray_config.ray_min_worker_port == 100

    kuscia_config.sf_cluster_desc = SFClusterDesc(
        parties=["alice", "bob"],
        devices=[
            SFClusterDesc.DeviceDesc(name="spu", type="SPU", parties=["alice", "bob"])
        ],
    )

    sf_cluster_config = get_sf_cluster_config(kuscia_config)

    logging.warning(f'sf_cluster_config: {sf_cluster_config}')
    assert list(sf_cluster_config.public_config.inference_config.parties) == [
        "alice",
        "bob",
    ]
    assert list(sf_cluster_config.public_config.inference_config.addresses) == [
        "0.0.0.0:1237",
        "http://bob_inference:1238",
    ]


def test_load_configs_with_table_attrs():
    kuscia_request_json = {
        'task_id': 'dwyk-lrhuiibc-node-3',
        'task_input_config': {
            'sf_datasource_config': {
                'bob': {'id': 'default-data-source'},
                'alice': {'id': 'default-data-source'},
            },
            'sf_cluster_desc': {
                'parties': ['bob', 'alice'],
                'devices': [
                    {
                        'name': 'spu',
                        'type': 'spu',
                        'parties': ['bob', 'alice'],
                        'config': '{"runtime_config":{"protocol":"SEMI2K","field":"FM128"},"link_desc":{"connect_retry_times":60,"connect_retry_interval_ms":1000,"brpc_channel_protocol":"http","brpc_channel_connection_type":"pooled","recv_timeout_ms":1200000,"http_timeout_ms":1200000}}',
                    },
                    {
                        'name': 'heu',
                        'type': 'heu',
                        'parties': ['bob', 'alice'],
                        'config': '{"mode": "PHEU", "schema": "paillier", "key_size": 2048}',
                    },
                ],
                'ray_fed_config': {'cross_silo_comm_backend': 'brpc_link'},
            },
            'sf_node_eval_param': {
                'comp_id': 'data_prep/psi:0.0.7',
                'attr_paths': [
                    'input/input_table_1/key',
                    'input/input_table_2/key',
                    'protocol',
                    'sort_result',
                    'allow_duplicate_keys',
                    'allow_duplicate_keys/no/skip_duplicates_check',
                    'ecdh_curve',
                    'allow_duplicate_keys/no/receiver_parties',
                ],
                'attrs': [
                    {'is_na': False, 'ss': ['id1']},
                    {'is_na': False, 'ss': ['id2']},
                    {'is_na': False, 's': 'PROTOCOL_RR22'},
                    {'b': True, 'is_na': False},
                    {'is_na': False, 's': 'no'},
                    {'is_na': True},
                    {'is_na': False, 's': 'CURVE_FOURQ'},
                    {'is_na': False, 'ss': ['alice', 'bob']},
                ],
                'inputs': [
                    {
                        'type': 'sf.table.individual',
                        'meta': {
                            '@type': 'type.googleapis.com/secretflow_spec.v1.IndividualTable',
                            'line_count': '-1',
                        },
                        'data_refs': [
                            {'uri': 'alice.csv', 'party': 'alice', 'format': 'csv'}
                        ],
                    },
                    {
                        'type': 'sf.table.individual',
                        'meta': {
                            '@type': 'type.googleapis.com/secretflow_spec.v1.IndividualTable',
                            'line_count': '-1',
                        },
                        'data_refs': [
                            {'uri': 'bob.csv', 'party': 'bob', 'format': 'csv'}
                        ],
                    },
                ],
                'checkpoint_uri': 'ckdwyk-lrhuiibc-node-3-output-0',
            },
            'sf_output_uris': ['dwyk_lrhuiibc_node_3_output_0'],
            'sf_input_ids': ['alice-table', 'bob-table'],
            'sf_input_partitions_spec': ['', ''],
            'sf_output_ids': ['dwyk-lrhuiibc-node-3-output-0'],
            'table_attrs': [
                {
                    'table_id': 'alice-table',
                    'column_attrs': [
                        {'col_name': 'id1', 'col_type': 'id'},
                        {'col_name': 'age', 'col_type': 'feature'},
                        {'col_name': 'education', 'col_type': 'feature'},
                        {'col_name': 'default', 'col_type': 'feature'},
                        {'col_name': 'balance', 'col_type': 'feature'},
                        {'col_name': 'housing', 'col_type': 'feature'},
                        {'col_name': 'loan', 'col_type': 'feature'},
                        {'col_name': 'day', 'col_type': 'feature'},
                        {'col_name': 'duration', 'col_type': 'feature'},
                        {'col_name': 'campaign', 'col_type': 'feature'},
                        {'col_name': 'pdays', 'col_type': 'feature'},
                        {'col_name': 'previous', 'col_type': 'feature'},
                        {'col_name': 'job_blue-collar', 'col_type': 'feature'},
                        {'col_name': 'job_entrepreneur', 'col_type': 'feature'},
                        {'col_name': 'job_housemaid', 'col_type': 'feature'},
                        {'col_name': 'job_management', 'col_type': 'feature'},
                        {'col_name': 'job_retired', 'col_type': 'feature'},
                        {'col_name': 'job_self-employed', 'col_type': 'feature'},
                        {'col_name': 'job_services', 'col_type': 'feature'},
                        {'col_name': 'job_student', 'col_type': 'feature'},
                        {'col_name': 'job_technician', 'col_type': 'feature'},
                        {'col_name': 'job_unemployed', 'col_type': 'feature'},
                        {'col_name': 'marital_divorced', 'col_type': 'feature'},
                        {'col_name': 'marital_married', 'col_type': 'feature'},
                        {'col_name': 'marital_single', 'col_type': 'feature'},
                    ],
                }
            ],
        },
        'task_cluster_def': '{"parties":[{"name":"bob", "role":"", "services":[{"portName":"spu", "endpoints":["dwyk-lrhuiibc-node-3-0-spu.bob.svc"]}, {"portName":"fed", "endpoints":["dwyk-lrhuiibc-node-3-0-fed.bob.svc"]}, {"portName":"global", "endpoints":["dwyk-lrhuiibc-node-3-0-global.bob.svc:31926"]}]}, {"name":"alice", "role":"", "services":[{"portName":"spu", "endpoints":["dwyk-lrhuiibc-node-3-0-spu.alice.svc"]}, {"portName":"fed", "endpoints":["dwyk-lrhuiibc-node-3-0-fed.alice.svc"]}, {"portName":"global", "endpoints":["dwyk-lrhuiibc-node-3-0-global.alice.svc:22915"]}]}], "selfPartyIdx":1, "selfEndpointIdx":0}',
        'allocated_ports': '{"ports":[{"name":"spu", "port":22919, "scope":"Cluster", "protocol":"GRPC"}, {"name":"fed", "port":22914, "scope":"Cluster", "protocol":"GRPC"}, {"name":"global", "port":22915, "scope":"Domain", "protocol":"GRPC"}, {"name":"node-manager", "port":22916, "scope":"Local", "protocol":"GRPC"}, {"name":"object-manager", "port":22917, "scope":"Local", "protocol":"GRPC"}, {"name":"client-server", "port":22918, "scope":"Local", "protocol":"GRPC"}]}',
    }

    kuscia_config = KusciaTaskConfig.from_json(kuscia_request_json)
    assert len(kuscia_config.table_attrs) == 1
    assert kuscia_config.table_attrs[0].table_id == "alice-table"
    assert len(kuscia_config.table_attrs[0].column_attrs) == 25
    assert kuscia_config.table_attrs[0].column_attrs[0].col_name == 'id1'
    assert kuscia_config.table_attrs[0].column_attrs[0].col_type == 'id'


def test_get_sf_cluster_config():
    sf_cluster_desc = SFClusterDesc(
        parties=["alice", "bob", "carol"],
        devices=[
            SFClusterDesc.DeviceDesc(name="spu", type="SPU", parties=["alice", "bob"])
        ],
    )

    kuscia_task_cluster_def = ClusterDefine(
        parties=[
            Party(
                name="alice",
                services=[
                    Service(port_name="fed", endpoints=["1.2.3.4"]),
                    Service(port_name="spu", endpoints=["1.2.3.4"]),
                    Service(port_name="global", endpoints=["0.0.0.0:1236"]),
                    Service(port_name="inference", endpoints=["0.0.0.0:1237"]),
                ],
            ),
            Party(
                name="bob",
                services=[
                    Service(port_name="fed", endpoints=["1.2.3.5"]),
                    Service(port_name="spu", endpoints=["1.2.3.5"]),
                    Service(port_name="inference", endpoints=["0.0.0.0:1238"]),
                ],
            ),
            Party(
                name="carol",
                services=[
                    Service(port_name="fed", endpoints=["1.2.3.6:2345"]),
                    Service(port_name="inference", endpoints=["0.0.0.0:1239"]),
                ],
            ),
        ],
        self_party_idx=0,
    )

    kuscia_task_allocated_ports = AllocatedPorts(
        ports=[
            Port(name="fed", port=1234),
            Port(name="spu", port=1235),
            Port(name="inference", port=1237),
        ]
    )

    kuscia_config = KusciaTaskConfig(
        task_id="task_id",
        task_cluster_def=kuscia_task_cluster_def,
        task_allocated_ports=kuscia_task_allocated_ports,
        sf_cluster_desc=sf_cluster_desc,
    )

    sf_cluster_config = get_sf_cluster_config(kuscia_config)

    logging.warning(f'sf_cluster_config: {sf_cluster_config}')
    assert list(sf_cluster_config.public_config.ray_fed_config.addresses) == [
        "0.0.0.0:1234",
        "1.2.3.5:80",
        "1.2.3.6:2345",
    ]

    assert list(sf_cluster_config.public_config.inference_config.addresses) == [
        "0.0.0.0:1237",
        "http://0.0.0.0:1238",
        "http://0.0.0.0:1239",
    ]

    assert list(sf_cluster_config.public_config.spu_configs[0].addresses) == [
        "0.0.0.0:1235",
        "http://1.2.3.5:80",
    ]

    assert sf_cluster_config.private_config.self_party == "alice"
    assert sf_cluster_config.private_config.ray_head_addr == "0.0.0.0:1236"


def test_convert_domain_data_to_individual_table():
    domain_data = DomainData(
        name="input",
        author="alice",
        relative_uri="x/y/z",
        type="table",
        columns=[
            DataColumn(name="f1", type="int"),
            DataColumn(name="f2", type="double", comment="feature"),
            DataColumn(name="id", type="str", comment="id"),
            DataColumn(name="label", type="int", comment="label"),
        ],
    )

    meta = IndividualTable(
        schema=TableSchema(
            features=["f1", "f2", "id", "label"],
            feature_types=["int", "double", "str", "int"],
        ),
        line_count=-1,
    )

    expected_individual_table = DistData(
        name="input",
        type="sf.table.individual",
        data_refs=[DistData.DataRef(uri="x/y/z", party="alice", format="csv")],
    )
    expected_individual_table.meta.Pack(meta)

    assert (
        convert_domain_data_to_individual_table(domain_data)
        == expected_individual_table
    )
