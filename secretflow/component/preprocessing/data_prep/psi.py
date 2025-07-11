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
from dataclasses import dataclass

import pandas as pd

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    PathCleanUp,
    Reporter,
    UnionGroup,
    VTable,
    VTableFormat,
    VTableParty,
    download_csv,
    register,
    upload_orc,
    uuid4,
)
from secretflow.component.preprocessing.data_prep.pis_utils import trans_keys_to_ids
from secretflow.device import PYU, reveal, wait
from secretflow.utils.errors import InvalidStateError


@dataclass
class LeftJoin:
    left_side: str = Field.party_attr(desc="Required for left join")


@dataclass
class JoinType(UnionGroup):
    inner_join: str = Field.selection_attr(desc="Inner join")
    left_join: LeftJoin = Field.struct_attr(desc="Left join")
    full_join: str = Field.selection_attr(desc="Full join")
    difference: str = Field.selection_attr(desc="Difference")


@dataclass
class ECDHProtocol(UnionGroup):
    CURVE_25519: str = Field.selection_attr(desc="CURVE_25519")
    CURVE_FOURQ: str = Field.selection_attr(desc="CURVE_FOURQ")
    CURVE_SM2: str = Field.selection_attr(desc="CURVE_SM2")
    CURVE_SECP256K1: str = Field.selection_attr(desc="CURVE_SECP256K1")


@dataclass
class Protocol(UnionGroup):
    PROTOCOL_ECDH: ECDHProtocol = Field.union_attr(
        desc="ECDH protocol.", default="CURVE_25519"
    )
    PROTOCOL_RR22: str = Field.selection_attr(desc="RR22 protocol.")
    PROTOCOL_KKRT: str = Field.selection_attr(desc="KKRT protocol.")


@register(domain="data_prep", version="1.0.0", name="psi")
class PSI(Component):
    '''
    PSI between two parties.
    '''

    protocol: Protocol = Field.union_attr(
        desc="PSI protocol.",
        default="PROTOCOL_RR22",
    )

    sort_result: bool = Field.attr(
        desc="If false, output is not promised to be aligned. Warning: disable this option may lead to errors in the following components. DO NOT TURN OFF if you want to append other components.",
        default=True,
    )
    receiver_parties: list[str] = Field.party_attr(
        desc="Party names of receiver for result, all party will be receivers default; if only one party receive result, the result will be single-party table, hence you can not connect it to component with union table input.",
        list_limit=Interval.closed(0, 2),
    )
    allow_empty_result: bool = Field.attr(
        desc="Whether to allow the result to be empty, if allowed, an empty file will be saved, if not, an error will be reported.",
        default=False,
    )
    join_type: JoinType = Field.union_attr(
        desc="join type, default is inner join.",
        default="inner_join",
    )

    input_ds1_keys_duplicated: bool = Field.attr(
        desc="Whether key columns have duplicated rows, default is True.",
        default=True,
    )
    input_ds1_keys: list[str] = Field.table_column_attr(
        "input_ds1",
        desc="Column(s) used to join.",
        limit=Interval.closed(1, None),
    )
    input_ds2_keys_duplicated: bool = Field.attr(
        desc="Whether key columns have duplicated rows, default is True.",
        default=True,
    )
    input_ds2_keys: list[str] = Field.table_column_attr(
        "input_ds2",
        desc="Column(s) used to join.",
        limit=Interval.closed(1, None),
    )
    input_ds1: Input = Field.input(
        desc="Individual table for party 1",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )
    input_ds2: Input = Field.input(
        desc="Individual table for party 2",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="Output vertical table",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    report: Output = Field.output(
        desc="Output psi report",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        tbl1 = VTable.from_distdata(self.input_ds1).get_party(0)
        tbl2 = VTable.from_distdata(self.input_ds2).get_party(0)

        tbl1.schema = trans_keys_to_ids(tbl1.schema, self.input_ds1_keys)
        tbl2.schema = trans_keys_to_ids(tbl2.schema, self.input_ds2_keys)
        input_tables = [tbl1, tbl2]

        broadcast_result = True

        receiver_party = tbl1.party
        if self.receiver_parties and len(self.receiver_parties) == 1:
            receiver_party = self.receiver_parties[0]
            broadcast_result = False

        join_type = self.join_type.get_selected()
        advanced_join_type = to_join_type(join_type)

        rand_id = uuid4(tbl1.party)
        root_dir = os.path.join(ctx.data_dir, rand_id)
        na_rep = rand_id

        keys = {
            tbl1.party: self.input_ds1_keys,
            tbl2.party: self.input_ds2_keys,
        }
        table_duplicated = {
            tbl1.party: self.input_ds1_keys_duplicated,
            tbl2.party: self.input_ds2_keys_duplicated,
        }

        ecdh_curve = "CURVE_25519"
        protocol = self.protocol.get_selected()
        if protocol == "PROTOCOL_ECDH":
            ecdh_curve = self.protocol.PROTOCOL_ECDH.get_selected()

        # build input and output paths
        receiver_tbl = tbl1 if receiver_party == tbl1.party else tbl2
        output_tables = input_tables if broadcast_result else [receiver_tbl]
        outout_csv_path = os.path.join(root_dir, f"{rand_id}_output.csv")

        output_paths = {}
        input_paths = {}
        for info in input_tables:
            if broadcast_result or receiver_party == info.party:
                output_paths[info.party] = outout_csv_path
            input_paths[info.party] = os.path.join(root_dir, info.uri)

        spu = ctx.make_spu()
        with PathCleanUp({tbl.party: root_dir for tbl in input_tables}):
            with ctx.trace_io():
                download_res = [
                    PYU(info.party)(download_csv)(
                        ctx.storage, info, input_paths[info.party], na_rep
                    )
                    for info in input_tables
                ]

                wait(download_res)
            with ctx.trace_running():
                psi_res = spu.psi(
                    keys=keys,
                    input_path=input_paths,
                    output_path=output_paths,
                    receiver=receiver_party,
                    broadcast_result=broadcast_result,
                    table_keys_duplicated=table_duplicated,
                    output_csv_na_rep=na_rep,
                    protocol=protocol,
                    ecdh_curve=ecdh_curve,
                    advanced_join_type=advanced_join_type,
                    left_side=(
                        self.join_type.left_join.left_side
                        if join_type == "left_join"
                        else ""
                    ),
                    disable_alignment=not self.sort_result,
                )

            with ctx.trace_io():
                output_uri = self.output_ds.uri
                upload_res = [
                    PYU(tbl.party)(upload_orc)(
                        ctx.storage,
                        output_uri,
                        output_paths[tbl.party],
                        tbl.schema,
                        na_rep,
                    )
                    for tbl in output_tables
                ]
                output_rows = reveal(upload_res)
                assert len(set(output_rows)) == 1
                output_rows = output_rows[0]

        if output_rows == 0 and not self.allow_empty_result:
            raise InvalidStateError(
                "Empty result is not allowed, please check your input data or set allow_empty_result to true."
            )

        system_info = self.input_ds1.system_info
        output_tables = [
            VTableParty(p.party, output_uri, str(VTableFormat.ORC), schema=p.schema)
            for p in output_tables
        ]
        output = VTable(output_uri, output_tables, output_rows, system_info)
        self.output_ds.data = output.to_distdata()

        report_tbl = pd.DataFrame(psi_res)

        report = Reporter("psi_report", "", system_info=system_info)
        report.add_tab(report_tbl)
        self.report.data = report.to_distdata()


def to_join_type(v: str) -> str:
    if v == "inner_join":
        return "JOIN_TYPE_INNER_JOIN"
    elif v == "left_join":
        return 'JOIN_TYPE_LEFT_JOIN'
    elif v == "full_join":
        return 'JOIN_TYPE_FULL_JOIN'
    elif v == "difference":
        return 'JOIN_TYPE_DIFFERENCE'
    else:
        raise ValueError(f"unknown join_type {v}")
