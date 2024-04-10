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


# 贡献者：tianlun + lyvansky
# lyvansky账号：lyvansky@163.com
import os
from typing import List

from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import (
    DistDataType,
    extract_distdata_info,
    extract_table_header,
    merge_individuals_to_vtable,
)
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    VerticalTable,
)

input_io_absent_enable = True
psi_10pc_comp = Component(
    "psi_10pc",
    domain="preprocessing",
    version="0.0.1",
    desc="PSI among multiple parties.",
    input_io_absent_enable=input_io_absent_enable,
)
psi_10pc_comp.str_attr(
    name="protocol",
    desc="PSI protocol.",
    is_list=False,
    is_optional=True,
    default_value="ECDH_PSI_NPC",
    allowed_values=["ECDH_PSI_NPC"],
)
psi_10pc_comp.int_attr(
    name="bucket_size",
    desc="Specify the hash bucket size used in PSI. Larger values consume more memory.",
    is_list=False,
    is_optional=True,
    default_value=1048576,
    lower_bound=0,
    lower_bound_inclusive=False,
)
psi_10pc_comp.str_attr(
    name="ecdh_curve_type",
    desc="Curve type for ECDH PSI.",
    is_list=False,
    is_optional=True,
    default_value="CURVE_FOURQ",
    allowed_values=["CURVE_25519", "CURVE_FOURQ", "CURVE_SM2", "CURVE_SECP256K1"],
)

psi_10pc_comp.io(
    io_type=IoType.INPUT,
    name="input_0",
    desc="Input sample individual table(0)",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join. If not provided, ids of the dataset will be used.",
        )
    ],
)
psi_10pc_comp.io(
    io_type=IoType.INPUT,
    name="input_1",
    desc="Input sample individual table(1)",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join. If not provided, ids of the dataset will be used.",
        )
    ],
)
psi_10pc_comp.io(
    io_type=IoType.INPUT,
    name="input_2",
    desc="Input sample individual table(2)",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join. If not provided, ids of the dataset will be used.",
        )
    ],
)
psi_10pc_comp.io(
    io_type=IoType.INPUT,
    name="input_3",
    desc="Input sample individual table(3)",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join. If not provided, ids of the dataset will be used.",
        )
    ],
)
psi_10pc_comp.io(
    io_type=IoType.INPUT,
    name="input_4",
    desc="Input sample individual table(4)",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join. If not provided, ids of the dataset will be used.",
        )
    ],
)
psi_10pc_comp.io(
    io_type=IoType.INPUT,
    name="input_5",
    desc="Input sample individual table(5)",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join. If not provided, ids of the dataset will be used.",
        )
    ],
)
psi_10pc_comp.io(
    io_type=IoType.INPUT,
    name="input_6",
    desc="Input sample individual table(6)",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join. If not provided, ids of the dataset will be used.",
        )
    ],
)
psi_10pc_comp.io(
    io_type=IoType.INPUT,
    name="input_7",
    desc="Input sample individual table(7)",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join. If not provided, ids of the dataset will be used.",
        )
    ],
)
psi_10pc_comp.io(
    io_type=IoType.INPUT,
    name="input_8",
    desc="Input sample individual table(8)",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join. If not provided, ids of the dataset will be used.",
        )
    ],
)
psi_10pc_comp.io(
    io_type=IoType.INPUT,
    name="input_9",
    desc="Input sample individual table(9)",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join. If not provided, ids of the dataset will be used.",
        )
    ],
)
psi_10pc_comp.io(
    io_type=IoType.OUTPUT,
    name="psi_output",
    desc="Output vertical table",
    types=[DistDataType.VERTICAL_TABLE],
)


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
    new_meta.num_lines = imeta.num_lines

    new_x.meta.Pack(new_meta)

    return new_x


@psi_10pc_comp.eval_fn
def multi_party_balanced_psi_eval_fn(
        *,
        ctx,
        protocol,
        bucket_size,
        ecdh_curve_type,
        input_0,
        input_0_key,
        input_1,
        input_1_key,
        input_2,
        input_2_key,
        input_3,
        input_3_key,
        input_4,
        input_4_key,
        input_5,
        input_5_key,
        input_6,
        input_6_key,
        input_7,
        input_7_key,
        input_8,
        input_8_key,
        input_9,
        input_9_key,
        psi_output,
):
    input_data_array = [input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9]
    input_key_array = [input_0_key, input_1_key, input_2_key, input_3_key, input_4_key, input_5_key, input_6_key,
                       input_7_key, input_8_key, input_9_key]

    input_0_path_format = extract_distdata_info(input_0)
    assert len(input_0_path_format) == 1
    input_0_party = list(input_0_path_format.keys())[0]

    # only local fs is supported at this moment.
    local_fs_wd = ctx.local_fs_wd

    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    import logging

    logging.warning(spu_config)
    key, input_path, output_path, data_refs, srcs, parties = input_params_process(input_data_array,
                                                                                  input_key_array,
                                                                                  local_fs_wd,
                                                                                  psi_output)

    cluster_def = spu_config["cluster_def"]
    nodes = []
    for node in cluster_def['nodes']:
        if node and node['party'] in parties:
            nodes.append(node)
    cluster_def['nodes'] = nodes

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    with ctx.tracer.trace_running():
        intersection_count = spu.psi_csv(
            key=key,
            input_path=input_path,
            output_path=output_path,
            receiver=input_0_party,
            sort=False,
            protocol=protocol,
            bucket_size=bucket_size,
            curve_type=ecdh_curve_type,
        )[0]["intersection_count"]

    output_db = DistData(
        name=psi_output,
        type=str(DistDataType.VERTICAL_TABLE),
        sys_info=input_0.sys_info,
        data_refs=data_refs,
    )

    output_db = merge_individuals_to_vtable(
        srcs,
        output_db,
    )
    vmeta = VerticalTable()
    assert output_db.meta.Unpack(vmeta)
    vmeta.num_lines = intersection_count
    output_db.meta.Pack(vmeta)

    return {"psi_output": output_db}


def input_params_process(input_data_arr, input_key_arr, local_fs_wd, psi_output):
    keys = {}
    input_path = {}
    output_path = {}
    data_refs = []
    srcs = []
    parties = []

    for data, key in zip(input_data_arr, input_key_arr):
        if data is not None:
            input_path_format = extract_distdata_info(data)
            input_party = list(input_path_format.keys())[0]
            input_pyu = PYU(input_party)
            parties.append(input_party)
            # If input_key is not provided, try to get input_key from ids of input_3ata.
            if len(key) == 0:
                key = list(extract_table_header(data, load_ids=True)[input_party].keys())
            keys[input_pyu] = key
            input_path[input_pyu] = os.path.join(
                local_fs_wd, input_path_format[input_party].uri
            )
            output_path[input_pyu] = os.path.join(local_fs_wd, psi_output)
            data_refs.append(DistData.DataRef(uri=psi_output, party=input_party, format="csv"))
            srcs.append(modify_schema(data, key))
    return keys, input_path, output_path, data_refs, srcs, parties
