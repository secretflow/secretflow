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

import json
import os
from typing import Dict, List, Tuple

import pandas as pd

from secretflow.component.core import DistDataType
from secretflow.component.test_framework.test_case import (
    ClusterCase,
    DAGInput,
    DataCase,
    NetCase,
    TestNode,
    TestVersion,
)
from secretflow.component.test_framework.test_controller import TestController


def test_node_from_dict(data) -> List[TestNode]:
    test_nodes = []

    for party, node_data in data.items():
        test_node = TestNode(
            party=party,
            local_fs_path=node_data['local_fs_path'],
            rayfed_port=node_data['rayfed_port'],
            spu_port=node_data['spu_port'],
            hostname=node_data['hostname'],
            docker_cpu_limit=node_data['docker_cpu_limit'],
            docker_mem_limit=node_data['docker_mem_limit'],
        )
        test_nodes.append(test_node)

    return test_nodes


def load_config_dict(path) -> Dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def test_version_from_dict(data) -> List[TestVersion]:
    test_versions = []

    for version_name, version_data in data.items():
        test_version = TestVersion(
            source=version_name,
            version=version_data.get("version", None),
            whl_paths=version_data.get("whl_paths", None),
        )
        test_versions.append(test_version)

    return test_versions


def net_case_from_dict(data) -> List[NetCase]:
    test_net_cases = []

    for net_case_name, net_case_data in data.items():
        test_net_case = NetCase(
            name=net_case_name,
            limit_mb=net_case_data.get("limit_mb", 0),
            limit_ms=net_case_data.get("limit_ms", 0),
        )
        test_net_cases.append(test_net_case)

    return test_net_cases


def cluster_case_from_dict(data) -> List[ClusterCase]:
    clusters = []
    for cluster_name, cluster_data in data.items():
        cluster = ClusterCase(
            name=cluster_name,
            protocol=cluster_data["protocol"],
            field=cluster_data["field"],
            fxp=cluster_data["fxp"],
            spu_parties=cluster_data["spu_parties"],
        )
        clusters.append(cluster)
    return clusters


def features_from_data_path_dict(
    path_dict, exclude_columns: Dict[str, List[str]], test_run: bool
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    features = {}
    feature_types = {}
    for key, path in path_dict.items():
        df = pd.read_csv(path, nrows=10)
        features_list = [f for f in df.columns if f not in exclude_columns.get(key, [])]
        feature_types_list = df[features_list].dtypes.tolist()
        feature_types_list = [str(f_type) for f_type in feature_types_list]
        if test_run:
            features_list = features_list[:10]
            feature_types_list = feature_types_list[:10]
        features[key] = features_list
        feature_types[key] = feature_types_list
    return features, feature_types


def data_cases_from_dict(data, test_run: bool) -> List[DataCase]:
    data_cases = []

    for data_case_name, data_case in data.items():
        dag_input_data_dict = {}
        for dag_input_name, dag_input_data in data_case.items():
            assert (
                dag_input_data["data_type"] == "sf.table.vertical_table"
            ), "only support vertical tables now."

            exclude_columns = {}
            for key, value in dag_input_data.get("label_columns").items():
                exclude_columns[key] = value

            if_cols = dag_input_data.get("id_columns", None)
            if if_cols is not None:
                for key, value in if_cols.items():
                    if key in exclude_columns:
                        exclude_columns[key] = exclude_columns[key] + value
                    else:
                        exclude_columns[key] = value
            features, feature_types = features_from_data_path_dict(
                dag_input_data["data_paths"], exclude_columns, test_run
            )
            dag_input = DAGInput(
                DistDataType(dag_input_data["data_type"]),
                dag_input_data["data_paths"],
                features,
                feature_types,
                dag_input_data.get("id_columns", None),
                dag_input_data.get("id_types", None),
                dag_input_data.get("label_columns"),
                dag_input_data.get("label_types"),
            )
            dag_input_data_dict[dag_input_name] = dag_input
        data_cases.append(DataCase(data_case_name, dag_input_data_dict))
    return data_cases


def recursive_fill(path_dict, current_dir):
    for k, val in path_dict.items():
        if isinstance(val, str):
            path_dict[k] = os.path.join(current_dir, val)
        else:
            path_dict[k] = recursive_fill(val, current_dir)
    return path_dict


def test_config_generation(config_dict: Dict, test_run: bool):
    test = TestController()

    # add node case
    test_node_config = load_config_dict(config_dict["node"])
    test_nodes = test_node_from_dict(test_node_config)
    for node in test_nodes:
        test.add_node(node)

    # add version case
    test_version_config = load_config_dict(config_dict["version"])
    test_versions = test_version_from_dict(test_version_config)
    for test_version in test_versions:
        test.add_test_version(test_version)

    # add net case
    net_case_config = load_config_dict(config_dict["version"])
    net_cases = net_case_from_dict(net_case_config)
    for net_case in net_cases:
        test.add_net_case(net_case)

    # add cluster case
    cluster_case_config = load_config_dict(config_dict["cluster"])
    cluster_cases = cluster_case_from_dict(cluster_case_config)
    for cluster_case in cluster_cases:
        test.add_cluster_case(cluster_case)

    # add dag inputs
    data_config = load_config_dict(config_dict["data"])
    data_cases = data_cases_from_dict(data_config, test_run)
    for data_case in data_cases:
        test.add_data_case(data_case)
    return test
