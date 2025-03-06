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

import importlib.resources as resources
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List

from pyarrow import csv, orc
from secretflow_spec.v1.data_pb2 import DistData

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Output,
    VTable,
    VTableField,
    VTableFieldKind,
    VTableParty,
    download_files,
    register,
    uuid4,
)
from secretflow.device import PYU, wait
from secretflow.utils.errors import InvalidArgumentError


@register(domain='ml.predict', version='1.1.0')
class ServingModelInferencer(Component):
    '''
    batch predicting online service models in offline
    '''

    receiver: str = Field.party_attr(
        desc="Party of receiver.",
    )
    pred_name: str = Field.attr(
        desc="Column name for predictions.",
        default="score",
    )
    input_block_size: int = Field.attr(
        desc="block size (Byte) for input data streaming",
        default=65536,
        minor_min=1,
    )
    serving_model: Input = Field.input(
        desc="Input serving model.",
        types=[DistDataType.SERVING_MODEL],
    )
    saved_columns: list[str] = Field.table_column_attr(
        "input_ds",
        desc="which columns should be saved with prediction result",
    )
    input_ds: Input = Field.input(
        desc="Input vertical table or individual table.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="Output prediction.",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        if not self.pred_name:
            raise InvalidArgumentError('pred_name cannot be empty')

        self_party = ctx.cluster_config.private_config.self_party
        receiver = self.receiver
        party_metas = init_party_metas(self.serving_model, self.input_ds)
        if receiver not in party_metas:
            raise InvalidArgumentError(
                "receiver should be in data_refs of the serving_model",
                detail={"receiver": receiver, "parties": list(party_metas.keys())},
            )
        input_vtable = VTable.from_distdata(self.input_ds)
        additional_col_names, id_column_name, pred_schema = get_col_infos(
            input_vtable,
            self_party,
            receiver,
            self.saved_columns,
            party_metas,
        )
        base_path = os.path.join(ctx.data_dir, self_party, "serving")
        output_pred_path = get_output_pred_path(base_path, self.output_ds.uri)
        download_serving_files(ctx, base_path, party_metas)
        party_metas = update_file_path_and_trans_orc_to_csv(party_metas, base_path)

        service_id = f"inference_service_{uuid4(receiver)}"
        inference_model_id = f"inference_model_{uuid4(receiver)}"
        inference_channel_protocol = "http"

        if self_party in party_metas:
            party_metas = update_party_metas(
                ctx.cluster_config.public_config.inference_config,
                party_metas,
            )

            party_serving_config = get_serving_config(
                party_metas, self_party, inference_model_id, inference_channel_protocol
            )
            party_inference_addresses = get_party_inference_addresses(party_metas)
            logging.debug(
                f"party_metas: {party_metas}, party_serving_config: {party_serving_config}, party_inference_addresses: {party_inference_addresses}"
            )

            serving_config_path = os.path.join(base_path, "serving.config")
            dump_serving_config(
                service_id,
                id_column_name,
                party_inference_addresses,
                base_path,
                serving_config_path,
                party_serving_config,
            )
            serving_config_option = '--serving_config_file=' + serving_config_path

            inference_config_path = os.path.join(base_path, "inference.config")
            create_dir_if_empty(output_pred_path)
            dump_inference_config(
                inference_config_path,
                self_party,
                receiver,
                output_pred_path,
                additional_col_names,
                self.pred_name,
                self.input_block_size,
            )
            inference_config_option = '--inference_config_file=' + inference_config_path

            try:
                with resources.path(
                    'secretflow_serving.tools.inferencer', 'inferencer'
                ) as binary_path:
                    subprocess.run(
                        [
                            str(binary_path),
                            serving_config_option,
                            inference_config_option,
                        ],
                        check=True,
                    )
            except FileNotFoundError as e:
                raise FileNotFoundError(f"File not found error: {e}")
            except ImportError as e:
                raise ImportError(f"Import error: {e}")
            except subprocess.CalledProcessError as e:
                raise subprocess.CalledProcessError(f"Inferencer execute error: {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error: {e}")

        receiver_pyu = PYU(receiver)

        def upload_pred_file(storage, local_fn, remote_fn):
            storage.upload_file(local_fn, remote_fn)

        wait(
            receiver_pyu(upload_pred_file)(
                ctx.storage, output_pred_path, self.output_ds.uri
            )
        )

        pred_schema.fields[self.pred_name] = VTableField(
            self.pred_name, "float32", VTableFieldKind.LABEL
        )
        receiver_meta = VTableParty(
            party=receiver,
            uri=self.output_ds.uri,
            format="csv",
            null_strs=[],
            schema=pred_schema,
        )
        vtable = VTable(
            name=self.output_ds.uri,
            parties=[receiver_meta],
            line_count=input_vtable.line_count,
        )
        self.output_ds.data = vtable.to_distdata()


def get_output_pred_path(base_path, user_input_path):
    abs_pred_path = os.path.abspath(os.path.join(base_path, user_input_path))
    # path can not contain ..
    assert abs_pred_path.startswith(
        base_path
    ), f'path: {user_input_path} contains .., which is unsafe'
    return abs_pred_path


@dataclass
class ServingConifg:
    id: str
    host: str
    communication_port: int
    channel_protocol: str
    model_id: str
    http_feature_source_port: int | None = None

    model_package_path: str | None = None
    file_path: str | None = None
    query_datas: List[str] | None = None
    query_context: str | None = None


@dataclass
class MetaInfo:
    model_package_path: str
    communication_port: str | None = None
    host: str | None = None
    file_path: str | None = None
    file_format: str | None = None


def create_dir_if_empty(path: str):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_col_infos(
    vtable: VTable,
    self_party: str,
    receiver: str,
    input_ds_saved_columns: List[str],
    party_metas: Dict[str, MetaInfo],
):
    if input_ds_saved_columns:
        parties = vtable.select(input_ds_saved_columns).parties
        assert (
            receiver in parties and len(parties) == 1
        ), "only columns from receiver can be saved"
        additional_col = input_ds_saved_columns
    else:
        additional_col = []

    vtable_id = vtable.select_by_kinds(VTableFieldKind.ID)
    self_id_col_name = (
        get_id_col_name(vtable_id, self_party) if self_party in party_metas else None
    )

    receiver_id_col_name = (
        self_id_col_name
        if receiver == self_party
        else get_id_col_name(vtable_id, receiver)
    )
    assert receiver_id_col_name is not None, "receiver doesn't have id col"

    additional_col = [col for col in additional_col if col != receiver_id_col_name]
    pred_schema = (
        vtable.select(additional_col + [receiver_id_col_name]).parties[receiver].schema
    )

    return (
        additional_col,
        self_id_col_name,
        pred_schema,
    )


def get_id_col_name(vtable_id: VTable, party: str):
    if party not in vtable_id.parties:
        return None

    kinds = vtable_id.parties[party].kinds
    id_col_names = [
        col_name for col_name, kind in kinds.items() if kind == VTableFieldKind.ID
    ]
    assert (
        len(id_col_names) == 1
    ), "serving model inferencer id col only support one id col for now "
    return id_col_names[0]


def download_serving_files(ctx, base_path: str, party_metas: Dict[str, MetaInfo]):
    download_path = {}
    remote_path = {}

    for party, meta in party_metas.items():
        download_path[party] = os.path.join(base_path, meta.model_package_path)
        remote_path[party] = meta.model_package_path

    with ctx.tracer.trace_io():
        download_files(ctx.storage, remote_path, download_path)

    for party, meta in party_metas.items():
        if meta.file_path:
            download_path[party] = os.path.join(base_path, meta.file_path)
            remote_path[party] = meta.file_path

    with ctx.tracer.trace_io():
        download_files(ctx.storage, remote_path, download_path)


def trans_orc_to_csv(orc_path: str, csv_path: str) -> str:
    try:
        orc_file = orc.ORCFile(orc_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read orc file: {orc_path}, error: {e}")

    if orc_file.nstripes == 0:
        raise RuntimeError(f"Empty orc file: {orc_path}")

    with csv.CSVWriter(csv_path, orc_file.schema) as csv_writer:
        for stripe in range(orc_file.nstripes):
            csv_writer.write(orc_file.read_stripe(stripe))


def update_file_path_and_trans_orc_to_csv(
    party_metas: Dict[str, MetaInfo], base_path: str
):
    waits = []
    for party, meta in party_metas.items():
        if meta.file_format == "orc":
            orc_path = os.path.join(base_path, meta.file_path)
            new_csv_path = orc_path + ".orc_to_csv.csv"
            waits.append(PYU(party)(trans_orc_to_csv)(orc_path, new_csv_path))
            party_metas[party].file_path = new_csv_path
    wait(waits)
    return party_metas


def init_party_metas(serving_model: DistData, dataset: DistData):
    party_metas = {}
    for data_ref in serving_model.data_refs:
        party_metas[data_ref.party] = MetaInfo(model_package_path=data_ref.uri)

    for data_ref in dataset.data_refs:
        if data_ref.party not in party_metas:
            continue
        assert (
            data_ref.format == "csv" or data_ref.format == "orc"
        ), "serving only support csv and orc for now"
        party_metas[data_ref.party].file_path = data_ref.uri
        party_metas[data_ref.party].file_format = data_ref.format

    return party_metas


def update_party_metas(inference_config, party_metas: dict[str, MetaInfo]):
    for i, party in enumerate(inference_config.parties):
        if party in party_metas.keys():
            host_port = inference_config.addresses[i].split(":")
            party_metas[party].host = host_port[0]
            if len(host_port) == 2:
                party_metas[party].communication_port = host_port[1]
    return party_metas


def get_serving_config(
    party_metas: dict[str, MetaInfo],
    self_party: str,
    model_id: str,
    channel_protocol: str,
):
    party_serving_config = ServingConifg(
        model_id=model_id,
        channel_protocol=channel_protocol,
        id=self_party,
        model_package_path=party_metas[self_party].model_package_path,
        file_path=party_metas[self_party].file_path,
        query_datas=[],
        query_context="",
        host=party_metas[self_party].host,
        communication_port=party_metas[self_party].communication_port,
    )
    return party_serving_config


def get_party_inference_addresses(party_metas: dict[str, MetaInfo]):
    parties = []
    for party, meta_info in party_metas.items():
        party_address = {}
        party_address["id"] = party
        party_address["address"] = meta_info.host
        if meta_info.communication_port is not None:
            party_address["address"] += ":" + meta_info.communication_port
        parties.append(party_address)
    return parties


def dump_json(obj, filename: str, indent=2):
    create_dir_if_empty(filename)
    with open(filename, "w") as ofile:
        json.dump(obj, ofile, indent=indent)


def dump_serving_config(
    service_id: str,
    id_column_name: str | None,
    parties: List[str],
    base_path: str,
    serving_config_path: str,
    config: ServingConifg,
):
    model_config_dict = {
        "modelId": config.model_id,
        "basePath": base_path,
        "sourcePath": os.path.join(base_path, config.model_package_path),
        "sourceType": "ST_FILE",
    }

    config_dict = {
        "id": service_id,
        "serverConf": {
            "host": config.host,
            "communicationPort": config.communication_port,
        },
        "modelConf": model_config_dict,
        "clusterConf": {
            "selfId": config.id,
            "parties": parties,
            "channel_desc": {
                "protocol": config.channel_protocol,
                "retryPolicyConfig": {
                    "retryCustom": "true",
                    "retryAggressive": "true",
                    "maxRetryCount": "3",
                    "fixedBackoffConfig": {"intervalMs": "100"},
                },
            },
        },
    }
    if id_column_name and config.file_path:
        config_dict["featureSourceConf"] = {
            "streamingOpts": {
                "file_path": os.path.join(base_path, config.file_path),
                "id_name": id_column_name,
            }
        }
    else:
        config_dict["featureSourceConf"] = {"mockOpts": {}}
    dump_json(config_dict, serving_config_path)


@dataclass
class InferenceConfig:
    requester_id: str = None
    # none for slave
    result_file_path: str = None
    # none for slave
    additional_col_names: List[str] = None
    score_col_name: str = None


def dump_inference_config(
    config_path: str,
    self_party: str,
    requester_id: str,
    result_file_path: str,
    additional_col_names: List[str],
    score_col_name: str,
    block_size: int,
):
    if self_party == requester_id:
        config_dict = {
            "requester_id": requester_id,
            "result_file_path": result_file_path,
            "additional_col_names": additional_col_names,
            "score_col_name": score_col_name,
            "block_size": block_size,
        }
    else:
        config_dict = {
            "requester_id": requester_id,
            "score_col_name": score_col_name,
            "block_size": block_size,
        }

    dump_json(config_dict, config_path)
