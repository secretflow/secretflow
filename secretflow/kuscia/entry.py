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

import logging
import os
import subprocess
import sys
from typing import List

import click
from google.protobuf import json_format
from kuscia.proto.api.v1alpha1.datamesh.domaindatasource_pb2 import DomainDataSource

from secretflow.component.entry import comp_eval, get_comp_def
from secretflow.kuscia.datamesh import (
    create_channel,
    create_domain_data_in_dm,
    create_domain_data_in_dp,
    create_domain_data_service_stub,
    create_domain_data_source_service_stub,
    get_csv_from_dp,
    create_dm_flight_client,
    get_domain_data,
    get_domain_data_source,
    put_data_to_dp,
)
from secretflow.kuscia.meta_conversion import (
    convert_dist_data_to_domain_data,
    convert_domain_data_to_individual_table,
)
from secretflow.kuscia.ray_config import RayConfig
from secretflow.kuscia.sf_config import get_sf_cluster_config
from secretflow.kuscia.task_config import KusciaTaskConfig
from secretflow.spec.v1.data_pb2 import DistData, StorageConfig
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam, NodeEvalResult

_LOG_FORMAT = "%(asctime)s|{}|%(levelname)s|secretflow|%(filename)s:%(funcName)s:%(lineno)d| %(message)s"

DEFAULT_DATAMESH_ADDRESS = "datamesh:8071"
TEMP_STORAGE_ROOT = "/tmp"


def start_ray(ray_conf: RayConfig):
    logging.info(f"ray_conf: {ray_conf}")

    ray_cmd = ray_conf.generate_ray_cmd()

    logging.info(
        f"Trying to start ray head node at {ray_conf.ray_node_ip_address}, start command: {ray_cmd}"
    )

    if not ray_cmd:
        # Local mode, do nothing here.
        return

    process = subprocess.run(ray_cmd, capture_output=True, shell=True)

    if process.returncode != 0:
        err_msg = f"Failed to start ray head node, start command: {ray_cmd}, stderr: {process.stderr}"
        logging.critical(err_msg)
        logging.critical("This process will exit now!")
        sys.exit(-1)
    else:
        if process.stdout:
            logging.info(process.stdout.decode(errors='ignore'))
        logging.info(
            f"Succeeded to start ray head node at {ray_conf.ray_node_ip_address}."
        )


def preprocess_sf_node_eval_param(
    task_conf: KusciaTaskConfig,
    datamesh_addr: str,
    param: NodeEvalParam,
    datasource: DomainDataSource,
    storage_config: StorageConfig,
    domaindata_stub,
    sf_input_ids: List[str] = None,
    sf_output_uris: List[str] = None,
) -> NodeEvalParam:
    comp_def = get_comp_def(param.domain, param.name, param.version)

    assert len(comp_def.inputs) == len(
        sf_input_ids
    ), "cnt of sf_input_ids doesn't match cnt of comp_def.inputs."
    assert len(comp_def.outputs) == len(
        sf_output_uris
    ), "cnt of sf_output_uris doesn't match cnt of comp_def.outputs."

    # get input DistData from GRM

    if len(sf_input_ids):
        if not datasource.access_directly:
            dm_flight_client = create_dm_flight_client(datamesh_addr)

        param.ClearField('inputs')
        for id, input_def in zip(sf_input_ids, list(comp_def.inputs)):
            domain_data = get_domain_data(domaindata_stub, id)

            if (
                domain_data.author == task_conf.party_name
                and domain_data.datasource_id != datasource.datasource_id
            ):
                raise RuntimeError(
                    f"datasource_id of domain_data [{domain_data.domaindata_id}] is {domain_data.datasource_id}, which doesn't match global datasource_id {datasource.datasource_id}"
                )

            if domain_data.attributes["dist_data"]:
                dist_data = json_format.Parse(
                    domain_data.attributes["dist_data"], DistData()
                )

            else:
                assert "sf.table.individual" in set(input_def.types)
                dist_data = convert_domain_data_to_individual_table(domain_data)

            param.inputs.append(dist_data)

            if not datasource.access_directly:
                assert dist_data.type in [
                    "sf.table.individual",
                    "sf.table.vertical_table",
                ], "only support tables."

                for data_ref in list(dist_data.data_refs):
                    if data_ref.party == task_conf.party_name:
                        get_csv_from_dp(
                            dm_flight_client,
                            id,
                            os.path.join(
                                storage_config.local_fs.wd, domain_data.relative_uri
                            ),
                        )

        if not datasource.access_directly:
            dm_flight_client.close()

    if len(sf_output_uris):
        param.ClearField('output_uris')
        param.output_uris.extend(sf_output_uris)

    return param


def get_datasource_id(task_conf: KusciaTaskConfig) -> str:
    party_name = task_conf.party_name
    sf_datasource_config = task_conf.sf_datasource_config
    if sf_datasource_config is not None:
        if party_name not in sf_datasource_config:
            raise RuntimeError(
                f"party {party_name} is missing in sf_datasource_config."
            )
        return sf_datasource_config[party_name]["id"]
    else:
        raise RuntimeError("sf_datasource_config is missing in task_conf.")


def get_storage_config(
    kuscia_config: KusciaTaskConfig,
    datasource: DomainDataSource,
) -> StorageConfig:
    if datasource.access_directly:
        storage_config = StorageConfig(
            type="local_fs",
            local_fs=StorageConfig.LocalFSConfig(wd=datasource.info.localfs.path),
        )
    else:
        default_localfs_path = os.path.join(
            TEMP_STORAGE_ROOT,
            f"sf_{kuscia_config.task_id}_{kuscia_config.party_name}",
        )

        if not os.path.exists(default_localfs_path):
            os.mkdir(default_localfs_path)

        storage_config = StorageConfig(
            type="local_fs",
            local_fs=StorageConfig.LocalFSConfig(wd=default_localfs_path),
        )

    return storage_config


def postprocess_sf_node_eval_result(
    task_conf: KusciaTaskConfig,
    res: NodeEvalResult,
    datasource: DomainDataSource,
    storage_config: StorageConfig,
    datamesh_addr: str,
    domaindata_stub,
    party: str,
    sf_output_ids: List[str] = None,
    sf_output_uris: List[str] = None,
) -> None:
    # write output DistData to GRM
    if sf_output_ids is not None and len(sf_output_ids) > 0:
        if not datasource.access_directly:
            dm_flight_client = create_dm_flight_client(datamesh_addr)

        for domain_data_id, dist_data, output_uri in zip(
            sf_output_ids, res.outputs, sf_output_uris
        ):
            domain_data = convert_dist_data_to_domain_data(
                domain_data_id, datasource.datasource_id, dist_data, output_uri, party
            )
            create_domain_data_in_dm(domaindata_stub, domain_data)
            if not datasource.access_directly:
                assert dist_data.type in [
                    "sf.table.individual",
                    "sf.table.vertical_table",
                ], "only support tables."

                for data_ref in list(dist_data.data_refs):
                    if data_ref.party == task_conf.party_name:
                        create_domain_data_in_dp(
                            dm_flight_client, datasource.datasource_id, domain_data
                        )
                        path = os.path.join(storage_config.local_fs.wd, data_ref.uri)
                        put_data_to_dp(dm_flight_client, domain_data_id, path)

        if not datasource.access_directly:
            dm_flight_client.close()


@click.command()
@click.argument("task_config_path", type=click.Path(exists=True))
@click.option("--datamesh_addr", required=False, default=DEFAULT_DATAMESH_ADDRESS)
def main(task_config_path, datamesh_addr):
    task_conf = KusciaTaskConfig.from_file(task_config_path)

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=_LOG_FORMAT.format(task_conf.party_name),
        force=True,
    )

    datamesh_channel = create_channel(datamesh_addr)

    datasource_stub = create_domain_data_source_service_stub(datamesh_channel)
    datasource_id = get_datasource_id(task_conf)
    datasource = get_domain_data_source(datasource_stub, datasource_id)

    ray_config = RayConfig.from_kuscia_task_config(task_conf)
    start_ray(ray_config)

    storage_config = get_storage_config(task_conf, datasource)

    domaindata_stub = create_domain_data_service_stub(datamesh_channel)
    sf_node_eval_param = preprocess_sf_node_eval_param(
        task_conf,
        datamesh_addr,
        task_conf.sf_node_eval_param,
        datasource,
        storage_config,
        domaindata_stub,
        task_conf.sf_input_ids,
        task_conf.sf_output_uris,
    )

    sf_cluster_config = get_sf_cluster_config(task_conf)

    res = comp_eval(sf_node_eval_param, storage_config, sf_cluster_config)

    postprocess_sf_node_eval_result(
        task_conf,
        res,
        datasource,
        storage_config,
        datamesh_addr,
        domaindata_stub,
        task_conf.party_name,
        task_conf.sf_output_ids,
        task_conf.sf_output_uris,
    )

    logging.info("Succeeded to run component.")

    sys.exit(0)


if __name__ == "__main__":
    main()
