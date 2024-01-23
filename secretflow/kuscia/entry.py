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

import base64
import logging
import os
import subprocess
import sys
import tarfile
from typing import List

import click
from google.protobuf import json_format
from kuscia.proto.api.v1alpha1.common_pb2 import FileFormat
from kuscia.proto.api.v1alpha1.datamesh.domaindatasource_pb2 import DomainDataSource

from secretflow.component.entry import comp_eval, get_comp_def
from secretflow.kuscia.datamesh import (
    create_channel,
    create_dm_flight_client,
    create_domain_data_in_dm,
    create_domain_data_in_dp,
    create_domain_data_service_stub,
    create_domain_data_source_service_stub,
    get_domain_data,
    get_domain_data_source,
    get_file_from_dp,
    put_file_to_dp,
)
from secretflow.kuscia.meta_conversion import (
    convert_dist_data_to_domain_data,
    convert_domain_data_to_individual_table,
)
from secretflow.kuscia.ray_config import RayConfig
from secretflow.kuscia.sf_config import get_sf_cluster_config
from secretflow.kuscia.task_config import KusciaTaskConfig
from secretflow.spec.v1.component_pb2 import IoDef
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


def download_dist_data_from_dp(
    task_conf: KusciaTaskConfig,
    dm_flight_client,
    domaindata_id,
    domain_data,
    storage_config,
    dist_data,
):
    # TODO: Refactor comp IO, move this to component/data_utils.py
    # IO should running in pyu context.
    data_type: str = dist_data.type
    need_untar = False
    party_data_refs = []
    for data_ref in list(dist_data.data_refs):
        if data_ref.party == task_conf.party_name:
            party_data_refs.append(data_ref)

    if data_type.startswith("sf.table"):
        # download csv, TODO: move to data_utils.py:load_table
        file_format = FileFormat.CSV
    elif data_type.startswith("sf.model") or data_type.startswith("sf.rule"):
        # download model etc., TODO: move to data_utils.py:model_loads
        file_format = FileFormat.BINARY
        need_untar = True
    elif data_type == "sf.serving.model":
        file_format = FileFormat.BINARY
    elif data_type in ["sf.report", "sf.read_data"]:
        # no data to download
        assert len(party_data_refs) == 0, "can not download report/read_data"
        return
    else:
        raise AttributeError(f"unknown dist_data type {data_type}")

    if party_data_refs:
        # FIXME: break this coupling
        path = os.path.join(storage_config.local_fs.wd, domain_data.relative_uri)
        if need_untar:
            path = f"{path}.tar.gz"
        get_file_from_dp(dm_flight_client, domaindata_id, path, file_format)
        if need_untar:
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(storage_config.local_fs.wd, filter='data')
        for data_ref in party_data_refs:
            file_path = os.path.join(storage_config.local_fs.wd, data_ref.uri)
            assert os.path.isfile(file_path), f"missing file {data_ref.uri}"


def domaindata_id_to_dist_data(
    task_conf: KusciaTaskConfig,
    dm_flight_client,
    datasource: DomainDataSource,
    storage_config: StorageConfig,
    domaindata_stub,
    domaindata_id,
    input_def: IoDef,
    skip_download_dataset: bool = False,
):
    domain_data = get_domain_data(domaindata_stub, domaindata_id)

    if (
        domain_data.author == task_conf.party_name
        and domain_data.datasource_id != datasource.datasource_id
    ):
        raise RuntimeError(
            f"datasource_id of domain_data [{domain_data.domaindata_id}] is {domain_data.datasource_id}, which doesn't match global datasource_id {datasource.datasource_id}"
        )

    if domain_data.attributes["dist_data"]:
        dist_data = json_format.Parse(domain_data.attributes["dist_data"], DistData())
        assert dist_data.type in set(input_def.types)
    else:
        assert "sf.table.individual" in set(input_def.types)
        dist_data = convert_domain_data_to_individual_table(domain_data)

    logging.info(f"domaindata_id {domaindata_id} to \n...........\n{dist_data}\n....")

    if not datasource.access_directly:
        if dist_data.type.startswith("sf.table") and skip_download_dataset:
            return dist_data

        download_dist_data_from_dp(
            task_conf,
            dm_flight_client,
            domaindata_id,
            domain_data,
            storage_config,
            dist_data,
        )

    return dist_data


def model_export_id_to_data(
    export_param: NodeEvalParam,
    task_conf: KusciaTaskConfig,
    dm_flight_client,
    datasource: DomainDataSource,
    storage_config: StorageConfig,
    domaindata_stub,
) -> NodeEvalParam:
    input_ids = None
    input_idx = None
    output_ids = None
    output_idx = None
    eval_params = None
    for i, path in enumerate(export_param.attr_paths):
        if path == "input_datasets":
            input_ids = list(export_param.attrs[i].ss)
            input_idx = i
        if path == "output_datasets":
            output_ids = list(export_param.attrs[i].ss)
            output_idx = i
        if path == "component_eval_params":
            eval_params = [
                json_format.Parse(base64.b64decode(p).decode("utf-8"), NodeEvalParam())
                for p in export_param.attrs[i].ss
            ]
    assert input_ids and output_ids and eval_params

    output_ds = []
    input_ds = []
    input_sum = 0
    output_sum = 0
    for param in eval_params:
        comp_def = get_comp_def(param.domain, param.name, param.version)
        assert input_sum + len(comp_def.inputs) <= len(input_ids)
        assert output_sum + len(comp_def.outputs) <= len(output_ids)

        for domain_id, input_def in zip(input_ids[input_sum:], list(comp_def.inputs)):
            input_ds.append(
                json_format.MessageToJson(
                    domaindata_id_to_dist_data(
                        task_conf,
                        dm_flight_client,
                        datasource,
                        storage_config,
                        domaindata_stub,
                        domain_id,
                        input_def,
                        skip_download_dataset=True,
                    ),
                    indent=0,
                )
            )
        for domain_id, output_def in zip(
            output_ids[output_sum:], list(comp_def.outputs)
        ):
            output_ds.append(
                json_format.MessageToJson(
                    domaindata_id_to_dist_data(
                        task_conf,
                        dm_flight_client,
                        datasource,
                        storage_config,
                        domaindata_stub,
                        domain_id,
                        output_def,
                        skip_download_dataset=True,
                    ),
                    indent=0,
                )
            )

        input_sum += len(comp_def.inputs)
        output_sum += len(comp_def.outputs)

    assert input_sum == len(input_ids)
    assert output_sum == len(output_ids)

    export_param.attrs[input_idx].ss[:] = input_ds
    export_param.attrs[output_idx].ss[:] = output_ds

    return export_param


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

    if not datasource.access_directly:
        dm_flight_client = create_dm_flight_client(datamesh_addr)
    else:
        dm_flight_client = None

    # get input DistData from GRM
    if len(sf_input_ids):
        param.ClearField('inputs')
        for domaindata_id, input_def in zip(sf_input_ids, list(comp_def.inputs)):
            param.inputs.append(
                domaindata_id_to_dist_data(
                    task_conf,
                    dm_flight_client,
                    datasource,
                    storage_config,
                    domaindata_stub,
                    domaindata_id,
                    input_def,
                )
            )

    if len(sf_output_uris):
        param.ClearField('output_uris')
        param.output_uris.extend(sf_output_uris)

    if param.domain == "model" and param.name == "model_export":
        # TODO: Refactor comp IO, unbind dataproxy/datamesh
        param = model_export_id_to_data(
            param,
            task_conf,
            dm_flight_client,
            datasource,
            storage_config,
            domaindata_stub,
        )

    if not datasource.access_directly:
        dm_flight_client.close()

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


def upload_dist_data_to_dp(
    task_conf: KusciaTaskConfig,
    dm_flight_client,
    domain_data_id,
    domain_data,
    storage_config,
    dist_data,
):
    # TODO: Refactor comp IO, move this to component/data_utils.py
    # IO should running in pyu context.
    data_type: str = dist_data.type
    party_data_refs = []
    for data_ref in list(dist_data.data_refs):
        if data_ref.party == task_conf.party_name:
            party_data_refs.append(data_ref)

    if data_type.startswith("sf.table"):
        # upload csv, TODO: move to data_utils.py:dump_vertical_table
        file_format = FileFormat.CSV
        if party_data_refs:
            assert len(party_data_refs) == 1
            data_ref = party_data_refs[0]
            # FIXME: break this coupling
            assert data_ref.uri == domain_data.relative_uri
            path = os.path.join(storage_config.local_fs.wd, data_ref.uri)
        else:
            path = None
    elif data_type.startswith("sf.model") or data_type.startswith("sf.rule"):
        # upload model etc., TODO: move to data_utils.py:model_dumps
        file_format = FileFormat.BINARY
        if party_data_refs:
            path = os.path.join(
                storage_config.local_fs.wd, f"{domain_data.relative_uri}.tar.gz"
            )
            with tarfile.open(path, "w:gz") as tar:
                for data_ref in party_data_refs:
                    # FIXME: break this coupling
                    assert data_ref.uri.startswith(domain_data.relative_uri)
                    tar.add(
                        os.path.join(storage_config.local_fs.wd, data_ref.uri),
                        arcname=data_ref.uri,
                    )
        else:
            path = None
    elif data_type == "sf.serving.model":
        # upload model etc., TODO: move to data_utils.py
        file_format = FileFormat.BINARY
        if party_data_refs:
            assert len(party_data_refs) == 1
            data_ref = party_data_refs[0]
            # FIXME: break this coupling
            assert data_ref.uri == domain_data.relative_uri
            path = os.path.join(storage_config.local_fs.wd, data_ref.uri)
        else:
            path = None
    elif data_type in ["sf.report", "sf.read_data"]:
        # no data to upload
        assert len(party_data_refs) == 0, "can not upload report/read_data"
        path = None
    else:
        raise AttributeError(f"unknown dist_data type {data_type}")

    if path:
        create_domain_data_in_dp(dm_flight_client, domain_data, file_format)
        put_file_to_dp(dm_flight_client, domain_data_id, path, file_format)


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
            logging.info(
                f"domaindata_id {domain_data_id} from \n...........\n{dist_data}\n...."
            )
            domain_data = convert_dist_data_to_domain_data(
                domain_data_id, datasource.datasource_id, dist_data, output_uri, party
            )
            create_domain_data_in_dm(domaindata_stub, domain_data)
            if not datasource.access_directly:
                upload_dist_data_to_dp(
                    task_conf,
                    dm_flight_client,
                    domain_data_id,
                    domain_data,
                    storage_config,
                    dist_data,
                )

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
    logging.info(
        f"datasource.access_directly {datasource.access_directly}\n"
        f"sf_node_eval_param  {json_format.MessageToJson(task_conf.sf_node_eval_param)} "
    )
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
