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
from secretflow.device import PYU, reveal
from secretflow.error_system.exceptions import CompEvalError


@dataclass
class AllowDuplicateKeysNo:
    skip_duplicates_check: bool = Field.attr(
        desc="If true, the check of duplicated items will be skiped.",
        default=False,
    )
    check_hash_digest: bool = Field.attr(
        desc="Check if hash digest of keys from parties are equal to determine whether to early-stop.",
        default=False,
    )
    receiver_parties: list[str] = Field.party_attr(
        desc="Party names of receiver for result, all party will be receivers default; if only one party receive result, the result will be single-party table, hence you can not connect it to component with union table input.",
        list_limit=Interval.closed(0, 2),
    )


@dataclass
class LeftJoin:
    left_side: str = Field.party_attr(desc="Required for left join")


@dataclass
class JoinType(UnionGroup):
    inner_join: str = Field.selection_attr(desc="Inner join with duplicate keys")
    left_join: LeftJoin = Field.struct_attr(desc="Left join with duplicate keys")
    full_join: str = Field.selection_attr(desc="Full join with duplicate keys")
    difference: str = Field.selection_attr(desc="Difference with duplicate keys")


@dataclass
class AllowDuplicateKeysYes:
    join_type: JoinType = Field.union_attr(desc="Join type.")


@dataclass
class AllowDuplicateKeys(UnionGroup):
    no: AllowDuplicateKeysNo = Field.struct_attr(desc="Duplicate keys are not allowed.")
    yes: AllowDuplicateKeysYes = Field.struct_attr(desc="Duplicate keys are allowed.")


@register(domain="data_prep", version="0.0.9", name="psi")
class PSI(Component):
    '''
    PSI between two parties.
    '''

    protocol: str = Field.attr(
        desc="PSI protocol.",
        choices=["PROTOCOL_RR22", "PROTOCOL_ECDH", "PROTOCOL_KKRT"],
        default="PROTOCOL_RR22",
    )

    sort_result: bool = Field.attr(
        desc="It false, output is not promised to be aligned. Warning: disable this option may lead to errors in the following components. DO NOT TURN OFF if you want to append other components.",
        default=True,
    )
    allow_empty_result: bool = Field.attr(
        desc="Whether to allow the result to be empty, if allowed, an empty file will be saved, if not, an error will be reported.",
        default=False,
    )
    allow_duplicate_keys: AllowDuplicateKeys = Field.union_attr(
        desc="Some join types allow duplicate keys. If you specify a party to receive, this should be no.",
    )
    ecdh_curve: str = Field.attr(
        desc="Curve type for ECDH PSI.",
        choices=["CURVE_25519", "CURVE_FOURQ", "CURVE_SM2", "CURVE_SECP256K1"],
        default="CURVE_FOURQ",
    )
    input_table_1_key: list[str] = Field.table_column_attr(
        "input_table_1",
        desc="Column(s) used to join.",
        limit=Interval.closed(1, None),
    )
    input_table_2_key: list[str] = Field.table_column_attr(
        "input_table_2",
        desc="Column(s) used to join.",
        limit=Interval.closed(1, None),
    )
    input_table_1: Input = Field.input(  # type: ignore
        desc="Individual table for party 1",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )
    input_table_2: Input = Field.input(  # type: ignore
        desc="Individual table for party 2",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )
    psi_output: Output = Field.output(
        desc="Output vertical table",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    report: Output = Field.output(
        desc="Output psi report",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        tbl1 = VTable.from_distdata(self.input_table_1).party(0)
        tbl2 = VTable.from_distdata(self.input_table_2).party(0)
        input_tables = [tbl1, tbl2]

        receiver_party = tbl1.party
        broadcast_result = True
        skip_duplicates_check = False
        check_hash_digest = False
        advanced_join_type = 'ADVANCED_JOIN_TYPE_UNSPECIFIED'
        left_side = "ROLE_RECEIVER"

        if self.allow_duplicate_keys.is_selected("no"):
            no = self.allow_duplicate_keys.no
            if len(no.receiver_parties) == 1:
                receiver_party = no.receiver_parties[0]
                broadcast_result = False
            skip_duplicates_check = no.skip_duplicates_check
            check_hash_digest = no.check_hash_digest
        else:
            yes = self.allow_duplicate_keys.yes
            join_type = yes.join_type.get_selected()
            left_side_party = yes.join_type.left_join.left_side
            advanced_join_type = to_join_type(join_type)
            if join_type == "left_join" and left_side_party != receiver_party:
                left_side = "ROLE_SENDER"

        rand_id = uuid4(tbl1.party)
        root_dir = os.path.join(ctx.data_dir, rand_id)
        na_rep = "NA"

        keys = {
            tbl1.party: self.input_table_1_key,
            tbl2.party: self.input_table_2_key,
        }

        # build input and output paths
        input_paths = {}
        receiver_tbl = tbl1 if receiver_party == tbl1.party else tbl2
        output_tables = input_tables if broadcast_result else [receiver_tbl]
        outout_csv_path = os.path.join(root_dir, f"{rand_id}_output.csv")
        output_paths = {}
        for info in input_tables:
            if broadcast_result or receiver_party == info.party:
                path = outout_csv_path
            else:
                path = ""

            output_paths[info.party] = path
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

                input_rows = reveal(download_res)

            with ctx.trace_running():
                spu.psi(
                    keys=keys,
                    input_path=input_paths,
                    output_path=output_paths,
                    receiver=receiver_party,
                    broadcast_result=broadcast_result,
                    protocol=self.protocol,
                    ecdh_curve=self.ecdh_curve,
                    advanced_join_type=advanced_join_type,
                    left_side=left_side,
                    skip_duplicates_check=skip_duplicates_check,
                    disable_alignment=not self.sort_result,
                    check_hash_digest=check_hash_digest,
                )

            with ctx.trace_io():
                output_uri = self.psi_output.uri
                upload_res = [
                    PYU(tbl.party)(upload_orc)(
                        ctx.storage,
                        output_uri,
                        output_paths[tbl.party],
                        tbl.schema.to_arrow(),
                        na_rep,
                    )
                    for tbl in output_tables
                ]
                output_rows = reveal(upload_res)
                assert len(set(output_rows)) == 1
                output_rows = output_rows[0]

        if output_rows == 0 and not self.allow_empty_result:
            raise CompEvalError(
                f"Empty result is not allowed, please check your input data or set allow_empty_result to true."
            )

        system_info = self.input_table_1.system_info
        output_tables = [
            VTableParty(p.party, output_uri, str(VTableFormat.ORC), schema=p.schema)
            for p in output_tables
        ]
        output = VTable(output_uri, output_tables, output_rows, system_info)
        self.psi_output.data = output.to_distdata()

        report_tbl = pd.DataFrame(
            {
                "party": [tbl.party for tbl in input_tables],
                "original_count": input_rows,
                "output_count": [output_rows for _ in range(2)],
            }
        )

        report = Reporter("psi_report", "")
        report.add_tab(report_tbl)
        report.dump_to(self.report, system_info)


def to_join_type(v: str) -> str:
    if v == "inner_join":
        return "ADVANCED_JOIN_TYPE_INNER_JOIN"
    elif v == "left_join":
        return 'ADVANCED_JOIN_TYPE_LEFT_JOIN'
    elif v == "full_join":
        return 'ADVANCED_JOIN_TYPE_FULL_JOIN'
    elif v == "difference":
        return 'ADVANCED_JOIN_TYPE_DIFFERENCE'
    else:
        raise ValueError(f"unknown join_type {v}")
