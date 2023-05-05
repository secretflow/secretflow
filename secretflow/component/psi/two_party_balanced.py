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


from secretflow.component.component import Component, IoType, TableColParam
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU
from secretflow.protos.component.comp_def_pb2 import TableType

two_party_balanced_psi_comp = Component(
    "two_party_balanced_psi",
    domain="psi",
    version="0.0.1",
    desc="Balanced PSI between two parties.",
)

two_party_balanced_psi_comp.str_param(
    name="receiver",
    desc="Which party can get joined data.",
    is_list=False,
    is_optional=True,
)
two_party_balanced_psi_comp.str_param(
    name="protocol",
    desc="PSI protocol.",
    is_list=False,
    is_optional=False,
    default_value="ECDH_PSI_2PC",
    allowed_values=["ECDH_PSI_2PC", "KKRT_PSI_2PC", "BC22_PSI_2PC"],
)
two_party_balanced_psi_comp.bool_param(
    name="precheck_input",
    desc="Whether to check input data before join.",
    is_list=False,
    is_optional=False,
    default_value=True,
)
two_party_balanced_psi_comp.bool_param(
    name="sort",
    desc="Whether sort data by key after join.",
    is_list=False,
    is_optional=False,
    default_value=True,
)
two_party_balanced_psi_comp.bool_param(
    name="broadcast_result",
    desc="Whether to broadcast joined data to all parties.",
    is_list=False,
    is_optional=False,
    default_value=True,
)
two_party_balanced_psi_comp.int_param(
    name="bucket_size",
    desc="Whether to broadcast joined data to all parties.",
    is_list=False,
    is_optional=False,
    default_value=1048576,
)
two_party_balanced_psi_comp.str_param(
    name="curve_type",
    desc="curve for ecdh psi.",
    is_list=False,
    is_optional=False,
    default_value="CURVE_FOURQ",
    allowed_values=["CURVE_25519", "CURVE_FOURQ", "CURVE_SM2", "CURVE_SECP256K1"],
)
two_party_balanced_psi_comp.table_io(
    io_type=IoType.INPUT,
    name="receiver_input",
    desc="input for receiver",
    types=[TableType.INDIVIDUAL_TABLE],
    col_params=[TableColParam(name="key", desc="Column(s) used to join.")],
)
two_party_balanced_psi_comp.table_io(
    io_type=IoType.INPUT,
    name="sender_input",
    desc="input for sender",
    types=[TableType.INDIVIDUAL_TABLE],
    col_params=[TableColParam(name="key", desc="Column(s) used to join.")],
)
two_party_balanced_psi_comp.table_io(
    io_type=IoType.OUTPUT,
    name="receiver_output",
    desc="output for receiver",
    types=[TableType.INDIVIDUAL_TABLE],
)
two_party_balanced_psi_comp.table_io(
    io_type=IoType.OUTPUT,
    name="sender_output",
    desc="output for sender",
    types=[TableType.INDIVIDUAL_TABLE],
)


@two_party_balanced_psi_comp.eval_fn
def two_party_balanced_psi_eval_fn(
    *,
    ctx,
    receiver,
    protocol,
    precheck_input,
    sort,
    broadcast_result,
    bucket_size,
    curve_type,
    receiver_input,
    sender_input,
    receiver_output,
    sender_output,
):
    receiver_input_path = receiver_input.table_metadata.indivial.path
    receiver_input_party = receiver_input.table_metadata.indivial.party
    for col_param in receiver_input.table_params.col_params:
        if col_param.name == "key":
            receiver_keys = list(col_param.cols)

    sender_input_path = sender_input.table_metadata.indivial.path
    sender_input_party = sender_input.table_metadata.indivial.party
    for col_param in sender_input.table_params.col_params:
        if col_param.name == "key":
            sender_keys = list(col_param.cols)

    receiver_output_path = receiver_output.table_metadata.indivial.path
    receiver_output_party = receiver_output.table_metadata.indivial.party

    sender_output_path = sender_output.table_metadata.indivial.path
    sender_output_party = sender_output.table_metadata.indivial.party

    spu = SPU(
        ctx['spu'],
        link_desc={
            'connect_retry_times': 60,
            'connect_retry_interval_ms': 1000,
            'brpc_channel_protocol': "http",
            "brpc_channel_connection_type": "pooled",
            'recv_timeout_ms': 1200 * 1000,  # 1200s
            'http_timeout_ms': 1200 * 1000,  # 1200s
        },
    )

    assert receiver_input_party == receiver_output_party
    assert sender_input_party == sender_output_party
    assert receiver_input_party != sender_input_party

    pyus = {k: PYU(k) for k in ctx['pyu']}

    assert receiver == receiver_input_party

    assert len(pyus) == 2
    assert receiver in pyus.keys()

    receiver_pyu = pyus[receiver]
    sender_pyu = pyus[sender_input_party]

    spu.psi_join_csv(
        key={receiver_pyu: receiver_keys, sender_pyu: sender_keys},
        input_path={
            receiver_pyu: receiver_input_path,
            sender_pyu: sender_input_path,
        },
        output_path={
            receiver_pyu: receiver_output_path,
            sender_pyu: sender_output_path,
        },
        receiver=receiver,
        join_party=sender_input_party,
        protocol=protocol,
        precheck_input=precheck_input,
        bucket_size=bucket_size,
        curve_type=curve_type,
    )
