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
import tarfile
from typing import List

import click
from dataproxy.sdk import FileFormat
from google.protobuf import json_format
from kuscia.proto.api.v1alpha1.datamesh.domaindatasource_pb2 import DomainDataSource

from secretflow.component.core import load_plugins
from secretflow.component.data_utils import DistDataType
from secretflow.component.entry import comp_eval, rebuild_comp_list
from secretflow.kuscia.datamesh import (
    create_channel,
    create_dm_flight_client,
    create_domain_data_in_dm,
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
from secretflow.kuscia.task_config import KusciaTaskConfig, TableAttr
from secretflow.spec.v1.data_pb2 import DistData, StorageConfig
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam, NodeEvalResult

_LOG_FORMAT = "%(asctime)s|{}|%(levelname)s|secretflow|%(filename)s:%(funcName)s:%(lineno)d| %(message)s"

DEFAULT_DATAMESH_ADDRESS = "datamesh:8071"
TEMP_STORAGE_ROOT = "/tmp"


def start_ray(ray_conf: RayConfig):
    logging.info(f"ray_conf: {ray_conf}")

    ray_cmd, envs = ray_conf.generate_ray_cmd()

    if not ray_cmd:
        # Local mode, do nothing here.
        return

    logging.info(
        f"Trying to start ray head node at {ray_conf.ray_node_ip_address}, start command: {' '.join(ray_cmd)}"
    )

    process = subprocess.run(ray_cmd, env=envs, capture_output=True, shell=False)

    if process.returncode != 0:
        err_msg = f"Failed to start ray head node, start command: {' '.join(ray_cmd)}, stderr: {process.stderr}"
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
    partition_spec: str = "",
):
    # TODO: Refactor comp IO, move this to component/data_utils.py
    # IO should running in pyu context.
    data_type: str = dist_data.type
    need_untar = False
    party_data_refs = []
    data_ref_type = set()
    for data_ref in list(dist_data.data_refs):
        if data_ref.party == task_conf.party_name:
            party_data_refs.append(data_ref)
            data_ref_type.add(data_ref.format)

    if not party_data_refs:
        return

    if data_type.startswith("sf.table"):
        if len(data_ref_type) != 1:
            raise RuntimeError(
                f"kuscia adapter load data refs of {task_conf.party_name} with multiple format: {data_ref_type}"
            )
        data_ref_type = data_ref_type.pop()
        # download csv, TODO: use DP SDK instead.
        if data_ref_type == 'csv':
            file_format = FileFormat.CSV
        elif data_ref_type == 'orc':
            file_format = FileFormat.ORC
        else:
            raise AttributeError(f"unrecognized format {data_ref_type} for {data_type}")

    elif data_type.startswith("sf.model") or data_type.startswith("sf.rule"):
        # download model etc., TODO: use DP SDK instead.
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
        get_file_from_dp(
            dm_flight_client, domaindata_id, path, file_format, partition_spec
        )
        if need_untar:
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(storage_config.local_fs.wd, filter='data')
        for data_ref in party_data_refs:
            file_path = os.path.join(storage_config.local_fs.wd, data_ref.uri)
            assert os.path.isfile(file_path), f"missing file {data_ref.uri}"


def get_table_info_from_table_attr(
    task_conf: KusciaTaskConfig, domaindata_id: str
) -> TableAttr:
    if not task_conf.table_attrs:
        logging.info(f"no table_attrs")
        return None
    for table_attr in task_conf.table_attrs:
        if table_attr.table_id == domaindata_id:
            return table_attr
    logging.info(f"domaindata_id {domaindata_id} not found in table_attrs")

    return None


def domaindata_id_to_dist_data(
    task_conf: KusciaTaskConfig,
    dm_flight_client,
    datasource: DomainDataSource,
    storage_config: StorageConfig,
    domaindata_stub,
    domaindata_id,
    partition_spec: str = "",
    skip_download_dataset: bool = False,
):
    if domaindata_id == '':
        return DistData(name='', type=str(DistDataType.NULL))

    domain_data = get_domain_data(domaindata_stub, domaindata_id)

    if (
        datasource.access_directly
        and domain_data.author == task_conf.party_name
        and domain_data.datasource_id != datasource.datasource_id
    ):
        raise RuntimeError(
            f"datasource_id of domain_data [{domain_data.domaindata_id}] is {domain_data.datasource_id}, which doesn't match global datasource_id {datasource.datasource_id}"
        )

    if domain_data.attributes["dist_data"]:
        dist_data = json_format.Parse(domain_data.attributes["dist_data"], DistData())
    else:
        table_attr = get_table_info_from_table_attr(task_conf, domaindata_id)
        dist_data = convert_domain_data_to_individual_table(domain_data, table_attr)

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
            partition_spec,
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
    for i, path in enumerate(export_param.attr_paths):
        if path == "input_datasets":
            input_ids = list(export_param.attrs[i].ss)
            input_idx = i
        if path == "output_datasets":
            output_ids = list(export_param.attrs[i].ss)
            output_idx = i
    assert input_ids and output_ids

    def _domain_ids_list_distdata_list(ids: list[str]) -> list[str]:
        result = []
        for domain_id in ids:
            dd = domaindata_id_to_dist_data(
                task_conf,
                dm_flight_client,
                datasource,
                storage_config,
                domaindata_stub,
                domain_id,
                partition_spec="",
                skip_download_dataset=True,
            )
            result.append(json_format.MessageToJson(dd, indent=0))

        return result

    input_ds = _domain_ids_list_distdata_list(input_ids)
    for domain_id in input_ids:
        input_ds.append(
            json_format.MessageToJson(
                domaindata_id_to_dist_data(
                    task_conf,
                    dm_flight_client,
                    datasource,
                    storage_config,
                    domaindata_stub,
                    domain_id,
                    partition_spec="",
                    skip_download_dataset=True,
                ),
                indent=0,
            )
        )
    output_ds = []
    for domain_id in output_ids:
        output_ds.append(
            json_format.MessageToJson(
                domaindata_id_to_dist_data(
                    task_conf,
                    dm_flight_client,
                    datasource,
                    storage_config,
                    domaindata_stub,
                    domain_id,
                    partition_spec="",
                    skip_download_dataset=True,
                ),
                indent=0,
            )
        )

    export_param.attrs[input_idx].ss[:] = _domain_ids_list_distdata_list(input_ids)
    export_param.attrs[output_idx].ss[:] = _domain_ids_list_distdata_list(output_ids)

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
    partitions_spec = task_conf.sf_input_partitions_spec
    if partitions_spec is not None and partitions_spec:
        assert len(partitions_spec) == len(
            sf_input_ids
        ), "cnt of sf_input_partions_spec doesn't match cnt of sf_input_ids."
    else:
        partitions_spec = [""] * len(sf_input_ids)

    output_partitions_spec = task_conf.sf_output_partitions_spec
    if output_partitions_spec is not None and output_partitions_spec:
        assert len(output_partitions_spec) == len(
            sf_output_uris
        ), "cnt of sf_output_partions_spec doesn't match cnt of sf_output_ids."

    if not datasource.access_directly:
        dm_flight_client = create_dm_flight_client(datamesh_addr)
    else:
        dm_flight_client = None

    # get input DistData from GRM
    if len(sf_input_ids):
        param.ClearField('inputs')
        for domaindata_id, partition_spec in zip(sf_input_ids, partitions_spec):
            param.inputs.append(
                domaindata_id_to_dist_data(
                    task_conf,
                    dm_flight_client,
                    datasource,
                    storage_config,
                    domaindata_stub,
                    domaindata_id,
                    partition_spec,
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
        if datasource.type == "localfs":
            storage_config = StorageConfig(
                type="local_fs",
                local_fs=StorageConfig.LocalFSConfig(wd=datasource.info.localfs.path),
            )
        elif datasource.type == "oss":
            storage_config = StorageConfig(
                type="s3",
                s3=StorageConfig.S3Config(
                    endpoint=datasource.info.oss.endpoint,
                    bucket=datasource.info.oss.bucket,
                    prefix=datasource.info.oss.prefix,
                    access_key_id=datasource.info.oss.access_key_id,
                    access_key_secret=datasource.info.oss.access_key_secret,
                    virtual_host=datasource.info.oss.virtualhost,
                    version=datasource.info.oss.version,
                ),
            )
        else:
            raise AttributeError(f"unsupported datasource.type {datasource.type}")
    else:
        # TODO: move DP io into comp storage.
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
        if party_data_refs:

            assert len(party_data_refs) == 1
            data_ref = party_data_refs[0]
            # FIXME: break this coupling
            assert data_ref.uri == domain_data.relative_uri
            path = os.path.join(storage_config.local_fs.wd, data_ref.uri)
            if data_ref.format == 'orc':
                file_format = FileFormat.ORC
            elif data_ref.format == 'csv':
                file_format = FileFormat.CSV
            else:
                raise AttributeError(
                    f"unsupported data ref format {data_ref.format} for {data_type}"
                )
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
        put_file_to_dp(dm_flight_client, domain_data_id, path, file_format, domain_data)


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
    # check valid for output partitions spec
    output_partitions_spec = task_conf.sf_output_partitions_spec
    if output_partitions_spec is not None and output_partitions_spec:
        assert len(output_partitions_spec) == len(
            sf_output_uris
        ), "cnt of sf_output_partions_spec doesn't match cnt of sf_output_ids."
    else:
        output_partitions_spec = [""] * len(sf_output_uris)
    # write output DistData to GRM
    if sf_output_ids is not None and len(sf_output_ids) > 0:
        if not datasource.access_directly:
            dm_flight_client = create_dm_flight_client(datamesh_addr)

        for domain_data_id, dist_data, output_uri, output_partition in zip(
            sf_output_ids, res.outputs, sf_output_uris, output_partitions_spec
        ):
            logging.info(
                f"domaindata_id {domain_data_id} from \n...........\n{dist_data}\n...."
            )
            domain_data = convert_dist_data_to_domain_data(
                domain_data_id,
                datasource.datasource_id,
                dist_data,
                output_uri,
                party,
                output_partition,
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
@click.option("--enable_plugins", is_flag=True, required=False, default=False)
def main(task_config_path, datamesh_addr, enable_plugins: bool):
    if enable_plugins:
        load_plugins()
        rebuild_comp_list()

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

    try:
        res = comp_eval(sf_node_eval_param, storage_config, sf_cluster_config)
    except Exception:
        logging.exception(f"comp_eval exception")
        os._exit(1)

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
    try:
        main()
    except Exception as e:
        logging.exception(f"unexpected exception")
        logging.shutdown()
        os._exit(1)
