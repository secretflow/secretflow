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
import copy
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Dict, List

import duckdb
import pyarrow as pa
from pyarrow import csv, orc

from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import (
    NP_DTYPE_TO_DUCKDB_DTYPE,
    DistDataType,
    download_files,
    extract_data_infos,
)
from secretflow.component.dataframe import StreamingReader, StreamingWriter
from secretflow.component.storage import ComponentStorage
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU
from secretflow.device.driver import reveal, wait
from secretflow.error_system.exceptions import CompEvalError, DataFormatError
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    StorageConfig,
    TableSchema,
)

psi_comp = Component(
    "psi",
    domain="data_prep",
    version="0.0.8",
    desc="PSI between two parties.",
)
psi_comp.str_attr(
    name="protocol",
    desc="PSI protocol.",
    is_list=False,
    is_optional=True,
    default_value="PROTOCOL_RR22",
    allowed_values=["PROTOCOL_RR22", "PROTOCOL_ECDH", "PROTOCOL_KKRT"],
)
psi_comp.bool_attr(
    name="sort_result",
    desc="It false, output is not promised to be aligned. Warning: disable this option may lead to errors in the following components. DO NOT TURN OFF if you want to append other components.",
    is_list=False,
    is_optional=True,
    default_value=True,
)
psi_comp.bool_attr(
    name="allow_empty_result",
    desc="Whether to allow the result to be empty, if allowed, an empty file will be saved, if not, an error will be reported.",
    is_list=False,
    is_optional=True,
    default_value=False,
)
psi_comp.union_attr_group(
    name="allow_duplicate_keys",
    desc="Some join types allow duplicate keys. If you specify a party to receive, this should be no.",
    group=[
        psi_comp.struct_attr_group(
            name="no",
            desc="Duplicate keys are not allowed.",
            group=[
                psi_comp.bool_attr(
                    name="skip_duplicates_check",
                    desc="If true, the check of duplicated items will be skiped.",
                    is_list=False,
                    is_optional=True,
                    default_value=False,
                ),
                psi_comp.bool_attr(
                    name="check_hash_digest",
                    desc="Check if hash digest of keys from parties are equal to determine whether to early-stop.",
                    is_list=False,
                    is_optional=True,
                    default_value=False,
                ),
                psi_comp.party_attr(
                    name="receiver_parties",
                    desc="Party names of receiver for result, all party will be receivers default; if only one party receive result, the result will be single-party table, hence you can not connect it to component with union table input.",
                    list_min_length_inclusive=0,
                    list_max_length_inclusive=2,
                ),
            ],
        ),
        psi_comp.struct_attr_group(
            name="yes",
            desc="Duplicate keys are allowed.",
            group=[
                psi_comp.union_attr_group(
                    name="join_type",
                    desc="Join type.",
                    group=[
                        psi_comp.union_selection_attr(
                            name="inner_join",
                            desc="Inner join with duplicate keys",
                        ),
                        psi_comp.struct_attr_group(
                            name="left_join",
                            desc="Left join with duplicate keys",
                            group=[
                                psi_comp.party_attr(
                                    name="left_side",
                                    desc="Required for left join",
                                    list_min_length_inclusive=1,
                                    list_max_length_inclusive=1,
                                )
                            ],
                        ),
                        psi_comp.union_selection_attr(
                            name="full_join",
                            desc="Full join with duplicate keys",
                        ),
                        psi_comp.union_selection_attr(
                            name="difference",
                            desc="Difference with duplicate keys",
                        ),
                    ],
                )
            ],
        ),
    ],
)

psi_comp.str_attr(
    name="ecdh_curve",
    desc="Curve type for ECDH PSI.",
    is_list=False,
    is_optional=True,
    default_value="CURVE_FOURQ",
    allowed_values=["CURVE_25519", "CURVE_FOURQ", "CURVE_SM2", "CURVE_SECP256K1"],
)


psi_comp.io(
    io_type=IoType.INPUT,
    name="input_table_1",
    desc="Individual table for party 1",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join.",
            col_min_cnt_inclusive=1,
        )
    ],
)
psi_comp.io(
    io_type=IoType.INPUT,
    name="input_table_2",
    desc="Individual table for party 2",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join.",
            col_min_cnt_inclusive=1,
        )
    ],
)
psi_comp.io(
    io_type=IoType.OUTPUT,
    name="psi_output",
    desc="Output vertical table",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
)


@dataclass
class PsiPartyInfo:
    party: str
    table: DistData
    keys: List[str]
    uri: str
    format: str


def trans_result_csv_to_orc(
    ctx,
    result_party_info: List[PsiPartyInfo],
    result_path: Dict[str, str],
    psi_output: str,
    allow_empty_result: bool,
) -> DistData:
    output_ctx = ctx
    if output_ctx.comp_storage._config.type.lower() != "local_fs":
        input_ctx = copy.deepcopy(output_ctx)
        input_ctx.comp_storage = ComponentStorage(
            StorageConfig(
                type="local_fs",
                local_fs=StorageConfig.LocalFSConfig(wd=output_ctx.data_dir),
            )
        )
    else:
        input_ctx = output_ctx

    streaming_infos = {}
    for party_info in result_party_info:
        streaming_infos.update(
            extract_data_infos(
                party_info.table, load_ids=True, load_features=True, load_labels=True
            )
        )

    for party, info in streaming_infos.items():
        info.uri = result_path[party]
        # TODO: PSI output type may be orc, change this later
        info.format = "csv"
        # for spu.psi config csv writer use NA replace null.
        if "NA" not in info.null_strs:
            info.null_strs.append("NA")
        # duckdb use NULL replace null.
        if "NULL" not in info.null_strs:
            info.null_strs.append("NULL")

    reader = StreamingReader.from_data_infos(
        input_ctx, streaming_infos, result_party_info[0].table.system_info
    )

    line = 0
    writer = StreamingWriter(output_ctx, psi_output, reader.data_infos)
    try:
        with writer:
            for batch in reader:
                writer.write(batch)
                line += batch.shape[1]
    except Exception as e:
        logging.error(f"Failed to write result {streaming_infos} to {psi_output}")
        raise e

    if line == 0 and not allow_empty_result:
        raise CompEvalError(
            f"Empty result is not allowed, please check your input data or set allow_empty_result to true."
        )

    return writer.to_distdata()


def add_keys_to_id_columns(table: DistData, keys: List[str]) -> DistData:
    assert table.type == DistDataType.INDIVIDUAL_TABLE
    meta = IndividualTable()
    if not table.meta.Unpack(meta):
        raise DataFormatError.unpack_distdata_error(unpack_type="IndividualTable")

    new_ids = []
    new_features = []
    found_keys = []
    for name, type in zip(meta.schema.features, meta.schema.feature_types):
        if name in keys:
            found_keys.append(name)
            new_ids.append((name, type))
        else:
            new_features.append((name, type))
    for name, type in zip(meta.schema.ids, meta.schema.id_types):
        if name in keys:
            found_keys.append(name)
            new_ids.append((name, type))
        else:
            # old id still in ids
            new_ids.append((name, type))

    if set(keys) != set(found_keys):
        raise RuntimeError(
            f"Keys {set(keys) - set(found_keys)} not found in table {table.name}"
        )

    new_meta = IndividualTable(
        schema=TableSchema(
            ids=[name for name, _ in new_ids],
            id_types=[type for _, type in new_ids],
            features=[name for name, _ in new_features],
            feature_types=[type for _, type in new_features],
            labels=meta.schema.labels,
            label_types=meta.schema.label_types,
        ),
        line_count=meta.line_count,
    )

    table.meta.Pack(new_meta)

    return table


def deal_null_from_csv(dist_data: DistData, old_dir: str, new_dir: str) -> str:
    assert (
        dist_data.type == DistDataType.INDIVIDUAL_TABLE
    ), f"{dist_data.type} is not individual table"

    table_info = extract_data_infos(
        dist_data, load_ids=True, load_features=True, load_labels=True
    )
    for party, info in table_info.items():
        duck_dtype = {c: NP_DTYPE_TO_DUCKDB_DTYPE[info.dtypes[c]] for c in info.dtypes}
        na_values = info.null_strs if info.null_strs else []

        with open(old_dir, 'rb') as io_reader:
            csv_db = duckdb.read_csv(io_reader, dtype=duck_dtype, na_values=na_values)
            col_list = [duckdb.ColumnExpression(c) for c in info.dtypes]
            csv_select = csv_db.select(*col_list)
            reader = csv_select.fetch_arrow_reader(batch_size=50000)
            pa.Table.from_pylist([], schema=reader.schema).to_pandas().to_csv(
                new_dir,
                index=False,
                mode='w',
                header=True,
            )
            num_rows = 0
            for batch in reader:
                num_rows += batch.num_rows
                batch_pd = batch.to_pandas()
                batch_pd.to_csv(
                    new_dir,
                    index=False,
                    mode='a',
                    header=False,
                    na_rep="NULL",
                )
            if num_rows == 0:
                raise RuntimeError(f"Party: {party}, Empty csv file: {info.uri}")


def get_psi_party_info(table: DistData, keys: List[str]) -> PsiPartyInfo:
    # TODO: delete this when upstream adds id & label info
    table = add_keys_to_id_columns(table, keys)

    def get_party_info_from_individual_table(input_table):
        path_info = extract_data_infos(input_table, load_ids=True)
        if len(path_info) == 0:
            raise RuntimeError(
                f"Individual table should have valid columns, {path_info}"
            )
        elif len(path_info) > 1:
            raise RuntimeError(
                f"Individual table should have only one data_ref, {path_info}"
            )
        else:
            return list(path_info.items())[0]

    party, party_info = get_party_info_from_individual_table(table)
    return PsiPartyInfo(
        party=party,
        table=table,
        keys=keys,
        uri=party_info.uri,
        format=party_info.format,
    )


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


def get_input_path(
    ctx, party_infos: List[PsiPartyInfo], task_id: str
) -> Dict[str, str]:
    download_path = {
        info.party: os.path.join(ctx.data_dir, task_id, info.uri)
        for info in party_infos
    }
    remote_path = {info.party: info.uri for info in party_infos}
    with ctx.tracer.trace_io():
        # TODO: avoid download file, support streaming in spu.psi
        download_files(ctx, remote_path, download_path)

    waits = []
    for info in party_infos:
        if info.format == "csv":
            continue
        elif info.format == "orc":
            # TODO: delete this when psi support orc
            new_path = download_path[info.party] + ".orc_to_csv.csv"
            waits.append(
                PYU(info.party)(trans_orc_to_csv)(download_path[info.party], new_path)
            )
            download_path[info.party] = new_path
        else:
            raise RuntimeError(f"Unsupported format {info.format}")

    wait(waits)

    new_path = {party: path + ".deal_null.csv" for party, path in download_path.items()}

    wait(
        [
            PYU(info.party)(deal_null_from_csv)(
                info.table, download_path[info.party], new_path[info.party]
            )
            for info in party_infos
        ]
    )

    return new_path


def make_output_path(result_path: Dict[str, str]):
    wait(
        [
            PYU(party)(lambda path: os.makedirs(path, exist_ok=True))(
                os.path.dirname(path)
            )
            for party, path in result_path.items()
        ]
    )


@psi_comp.eval_fn
def two_party_balanced_psi_eval_fn(
    *,
    ctx,
    protocol,
    sort_result,
    allow_empty_result,
    ecdh_curve,
    allow_duplicate_keys,
    allow_duplicate_keys_no_skip_duplicates_check,
    allow_duplicate_keys_no_check_hash_digest,
    allow_duplicate_keys_yes_join_type,
    allow_duplicate_keys_yes_join_type_left_join_left_side,
    allow_duplicate_keys_no_receiver_parties,
    input_table_1,
    input_table_1_key,
    input_table_2,
    input_table_2_key,
    psi_output,
):
    assert allow_duplicate_keys in [
        'yes',
        'no',
    ]

    broadcast_result = False
    if allow_duplicate_keys == 'yes':
        if len(allow_duplicate_keys_no_receiver_parties) not in (0, 2):
            raise CompEvalError.party_check_failed(
                f"allow_duplicate_keys_no_receiver_parties should be empty or have two parties, {allow_duplicate_keys_no_receiver_parties}"
            )
        broadcast_result = True
    elif (
        len(allow_duplicate_keys_no_receiver_parties) == 0
        or len(allow_duplicate_keys_no_receiver_parties) == 2
    ):
        broadcast_result = True

    sender_info = get_psi_party_info(input_table_1, input_table_1_key)
    receiver_info = get_psi_party_info(input_table_2, input_table_2_key)

    if (
        allow_duplicate_keys == 'no'
        and len(allow_duplicate_keys_no_receiver_parties) == 1
        and allow_duplicate_keys_no_receiver_parties[0] != receiver_info.party
    ):
        sender_info, receiver_info = receiver_info, sender_info

    if allow_duplicate_keys == 'no':
        advanced_join_type = 'ADVANCED_JOIN_TYPE_UNSPECIFIED'
    else:
        allow_duplicate_keys_no_skip_duplicates_check = False
        allow_duplicate_keys_no_check_hash_digest = False
        assert allow_duplicate_keys_yes_join_type in [
            "inner_join",
            "left_join",
            "full_join",
            "difference",
        ]

        if allow_duplicate_keys_yes_join_type == "inner_join":
            advanced_join_type = "ADVANCED_JOIN_TYPE_INNER_JOIN"
        elif allow_duplicate_keys_yes_join_type == "left_join":
            advanced_join_type = 'ADVANCED_JOIN_TYPE_LEFT_JOIN'
        elif allow_duplicate_keys_yes_join_type == "full_join":
            advanced_join_type = 'ADVANCED_JOIN_TYPE_FULL_JOIN'
        else:
            advanced_join_type = 'ADVANCED_JOIN_TYPE_DIFFERENCE'

    if advanced_join_type == 'ADVANCED_JOIN_TYPE_LEFT_JOIN':
        left_side = allow_duplicate_keys_yes_join_type_left_join_left_side[0]

        assert left_side in [
            sender_info.party,
            receiver_info.party,
        ]
    else:
        left_side = receiver_info.party

    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))
    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    task_id = reveal(PYU(sender_info.party)(lambda: str(uuid.uuid4()))())
    input_path = get_input_path(ctx, [receiver_info, sender_info], task_id)

    if broadcast_result:
        result_party_infos = [sender_info, receiver_info]
    else:
        result_party_infos = [receiver_info]

    output_csv_filename = f"{psi_output}.csv"
    result_path = {
        party_info.party: os.path.join(ctx.data_dir, task_id, output_csv_filename)
        for party_info in result_party_infos
    }
    make_output_path(result_path)

    output_path_stub = {
        party_info.party: "" for party_info in [receiver_info, sender_info]
    }
    output_path_stub.update(result_path)
    with ctx.tracer.trace_running():
        spu.psi(
            keys={
                receiver_info.party: receiver_info.keys,
                sender_info.party: sender_info.keys,
            },
            input_path=input_path,
            output_path=output_path_stub,
            receiver=receiver_info.party,
            broadcast_result=broadcast_result,
            protocol=protocol,
            ecdh_curve=ecdh_curve,
            advanced_join_type=advanced_join_type,
            left_side=(
                'ROLE_RECEIVER' if left_side == receiver_info.party else 'ROLE_SENDER'
            ),
            skip_duplicates_check=allow_duplicate_keys_no_skip_duplicates_check,
            disable_alignment=not sort_result,
            check_hash_digest=allow_duplicate_keys_no_check_hash_digest,
        )

    with ctx.tracer.trace_io():
        output_db = trans_result_csv_to_orc(
            ctx,
            result_party_infos,
            result_path,
            psi_output,
            allow_empty_result,
        )

    return {"psi_output": output_db}
