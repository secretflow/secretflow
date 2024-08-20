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
import os
from typing import Dict, List, Union

import pandas as pd
from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import (
    SUPPORTED_VTABLE_DATA_TYPE,
    DistDataType,
    download_files,
    extract_distdata_info,
    merge_individuals_to_vtable,
    upload_files,
)
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU
from secretflow.device.driver import wait
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable, VerticalTable

psi_tp_comp = Component(
    "psi_tp",
    domain="data_prep",
    version="0.0.5",
    desc="PSI between three parties.",
)
psi_tp_comp.str_attr(
    name="protocol",
    desc="PSI protocol.",
    is_list=False,
    is_optional=True,
    default_value="ECDH_PSI_3PC",
    allowed_values=["ECDH_PSI_3PC"],
)
psi_tp_comp.int_attr(
    name="fill_value_int",
    desc="For int type data. Use this value for filling null.",
    is_list=False,
    is_optional=True,
    default_value=0,
)

psi_tp_comp.str_attr(
    name="ecdh_curve",
    desc="Curve type for ECDH PSI.",
    is_list=False,
    is_optional=True,
    default_value="CURVE_FOURQ",
    allowed_values=["CURVE_25519", "CURVE_FOURQ", "CURVE_SM2", "CURVE_SECP256K1"],
)
psi_tp_comp.io(
    io_type=IoType.INPUT,
    name="receiver_input",
    desc="Individual table for receiver",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join.",
            col_min_cnt_inclusive=1,
        )
    ],
)
psi_tp_comp.io(
    io_type=IoType.INPUT,
    name="sender1_input",
    desc="Individual table for sender1",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join.",
            col_min_cnt_inclusive=1,
        )
    ],
)
psi_tp_comp.io(
    io_type=IoType.INPUT,
    name="sender2_input",
    desc="Individual table for sender2",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join.",
            col_min_cnt_inclusive=1,
        )
    ],
)
psi_tp_comp.io(
    io_type=IoType.OUTPUT,
    name="psi_output",
    desc="Output vertical table",
    types=[DistDataType.VERTICAL_TABLE],
)


def convert_int(x, fill_value_int, int_type_str):
    try:
        return SUPPORTED_VTABLE_DATA_TYPE[int_type_str](x)
    except Exception:
        return fill_value_int


def build_converters(x: DistData, fill_value_int: int) -> Dict[str, callable]:
    if x.type != "sf.table.individual":
        raise CompEvalError("Only support individual table")
    imeta = IndividualTable()
    assert x.meta.Unpack(imeta)
    converters = {}

    def assign_converter(i, t):
        if "int" in t:
            converters[i] = lambda x: convert_int(x, fill_value_int, t)

    for i, t in zip(list(imeta.schema.ids), list(imeta.schema.id_types)):
        assign_converter(i, t)

    for i, t in zip(list(imeta.schema.features), list(imeta.schema.feature_types)):
        assign_converter(i, t)

    for i, t in zip(list(imeta.schema.labels), list(imeta.schema.label_types)):
        assign_converter(i, t)

    return converters


# We would respect user-specified ids even ids are set in TableSchema.
def modify_schema(x: DistData, keys: List[str]) -> DistData:
    new_x = DistData()
    new_x.CopyFrom(x)
    if len(keys) == 0:
        return new_x
    assert x.type == "sf.table.individual"
    imeta = IndividualTable()
    assert x.meta.Unpack(imeta)

    new_meta = IndividualTable()
    names = []
    types = []

    # copy current ids to features and clean current ids.
    for i, t in zip(list(imeta.schema.ids), list(imeta.schema.id_types)):
        names.append(i)
        types.append(t)

    for f, t in zip(list(imeta.schema.features), list(imeta.schema.feature_types)):
        names.append(f)
        types.append(t)

    for k in keys:
        if k not in names:
            raise CompEvalError(f"key {k} is not found as id or feature.")

    for n, t in zip(names, types):
        if n in keys:
            new_meta.schema.ids.append(n)
            new_meta.schema.id_types.append(t)
        else:
            new_meta.schema.features.append(n)
            new_meta.schema.feature_types.append(t)

    new_meta.schema.labels.extend(list(imeta.schema.labels))
    new_meta.schema.label_types.extend(list(imeta.schema.label_types))
    new_meta.line_count = imeta.line_count

    new_x.meta.Pack(new_meta)

    return new_x


def read_fillna_write(
    csv_file_path: str,
    converters: Dict[str, callable],
    chunksize: int = 50000,
):
    # Define the CSV reading in chunks
    csv_chunks = pd.read_csv(csv_file_path, converters=converters, chunksize=chunksize)
    temp_file_path = csv_file_path + '.tmp'
    # Process each chunk
    for i, chunk in enumerate(csv_chunks):
        # Write the first chunk with headers, subsequent chunks without headers
        if i == 0:
            chunk.to_csv(temp_file_path, index=False)
        else:
            chunk.to_csv(temp_file_path, mode='a', header=False, index=False)
    # Replace the original file with the processed file
    os.replace(temp_file_path, csv_file_path)


def fill_missing_values(
    local_fns: Dict[Union[str, PYU], str],
    partywise_converters=Dict[str, Dict[str, callable]],
):
    pyu_locals = {p.party if isinstance(p, PYU) else p: local_fns[p] for p in local_fns}

    waits = []
    for p in pyu_locals:
        waits.append(PYU(p)(read_fillna_write)(pyu_locals[p], partywise_converters[p]))
    wait(waits)


@psi_tp_comp.eval_fn
def two_party_balanced_psi_eval_fn(
    *,
    ctx,
    protocol,
    ecdh_curve,
    fill_value_int,
    receiver_input,
    receiver_input_key,
    sender1_input,
    sender1_input_key,
    sender2_input,
    sender2_input_key,
    psi_output,
):
    receiver_path_format = extract_distdata_info(receiver_input)
    assert len(receiver_path_format) == 1
    receiver_party = list(receiver_path_format.keys())[0]
    sender1_path_format = extract_distdata_info(sender1_input)
    sender1_party = list(sender1_path_format.keys())[0]
    sender2_path_format = extract_distdata_info(sender2_input)
    sender2_party = list(sender2_path_format.keys())[0]

    receiver_pyu = PYU(receiver_party)
    sender1_pyu = PYU(sender1_party)
    sender2_pyu = PYU(sender2_party)

    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    input_path = {
        receiver_pyu: os.path.join(
            ctx.data_dir, receiver_path_format[receiver_party].uri
        ),
        sender1_pyu: os.path.join(ctx.data_dir, sender1_path_format[sender1_party].uri),
        sender2_pyu: os.path.join(ctx.data_dir, sender2_path_format[sender2_party].uri),
    }
    output_path = {
        receiver_pyu: os.path.join(ctx.data_dir, psi_output),
        sender1_pyu: os.path.join(ctx.data_dir, psi_output),
        sender2_pyu: os.path.join(ctx.data_dir, psi_output),
    }

    import logging

    logging.warning(spu_config)

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    uri = {
        receiver_party: receiver_path_format[receiver_party].uri,
        sender1_party: sender1_path_format[sender1_party].uri,
        sender2_party: sender2_path_format[sender2_party].uri,
    }

    with ctx.tracer.trace_io():
        download_files(ctx, uri, input_path)

    with ctx.tracer.trace_running():
        report = spu.psi_csv(
            key={receiver_pyu: receiver_input_key, sender1_pyu: sender1_input_key, sender2_pyu: sender2_input_key},
            input_path=input_path,
            output_path=output_path,
            receiver=receiver_party,
            broadcast_result=True,
            protocol=protocol,
            curve_type=ecdh_curve,
        )

    partywise_converters = {
        receiver_party: build_converters(receiver_input, fill_value_int),
        sender1_party: build_converters(sender1_input, fill_value_int),
        sender2_party: build_converters(sender2_input, fill_value_int),
    }

    with ctx.tracer.trace_io():
        fill_missing_values(output_path, partywise_converters)
        upload_files(
            ctx, {receiver_party: psi_output, sender1_party: psi_output, sender2_party: psi_output}, output_path
        )

    output_db = DistData(
        name=psi_output,
        type=str(DistDataType.VERTICAL_TABLE),
        system_info=receiver_input.system_info,
        data_refs=[
            DistData.DataRef(
                uri=psi_output,
                party=receiver_party,
                format="csv",
            ),
            DistData.DataRef(
                uri=psi_output,
                party=sender1_party,
                format="csv",
            ),
            DistData.DataRef(
                uri=psi_output,
                party=sender2_party,
                format="csv",
            ),
        ],
    )

    output_db = merge_individuals_to_vtable(
        [
            modify_schema(receiver_input, receiver_input_key),
            modify_schema(sender1_input, sender1_input_key),
            modify_schema(sender2_input, sender2_input_key),
        ],
        output_db,
    )
    vmeta = VerticalTable()
    assert output_db.meta.Unpack(vmeta)
    vmeta.line_count = report[0]['intersection_count']
    output_db.meta.Pack(vmeta)

    return {"psi_output": output_db}
