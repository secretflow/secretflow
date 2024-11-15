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

import os

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    VTable,
    download_csv,
    register,
    upload_orc,
    uuid4,
)
from secretflow.component.core.utils import PathCleanUp
from secretflow.device import PYU, reveal, wait


@register(domain="data_prep", version="1.0.0", name="psi_tp")
class PSIThreeParty(Component):
    '''
    PSI between three parties.
    '''

    ecdh_curve: str = Field.attr(
        desc="Curve type for ECDH PSI.",
        choices=["CURVE_FOURQ", "CURVE_25519", "CURVE_SM2", "CURVE_SECP256K1"],
        default="CURVE_FOURQ",
    )
    keys1: list[str] = Field.table_column_attr(
        "input_ds1",
        desc="Column(s) used to join.",
        limit=Interval.closed(1, None),
    )
    keys2: list[str] = Field.table_column_attr(
        "input_ds2",
        desc="Column(s) used to join.",
        limit=Interval.closed(1, None),
    )
    keys3: list[str] = Field.table_column_attr(
        "input_ds3",
        desc="Column(s) used to join.",
        limit=Interval.closed(1, None),
    )
    input_ds1: Input = Field.input(  # type: ignore
        desc="Individual table for party 1",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )
    input_ds2: Input = Field.input(  # type: ignore
        desc="Individual table for party 2",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )
    input_ds3: Input = Field.input(  # type: ignore
        desc="Individual table for party 3",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )

    output_ds: Output = Field.output(
        desc="Output vertical table",
        types=[DistDataType.VERTICAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        inputs = [
            VTable.from_distdata(self.input_ds1).party(0),
            VTable.from_distdata(self.input_ds2).party(0),
            VTable.from_distdata(self.input_ds3).party(0),
        ]
        keys_list = [self.keys1, self.keys2, self.keys3]
        input_parties = {t.party: t for t in inputs}

        uuid = uuid4(inputs[0].party)
        root_dir = os.path.join(ctx.data_dir, uuid)
        output_csv_path = os.path.join(root_dir, f"{uuid}_output.csv")

        na_rep = "NA"
        receiver = inputs[0].party
        psi_keys = {}
        psi_input_paths = {}
        psi_output_paths = {}
        for p, keys in zip(inputs, keys_list):
            if keys not in p.schema:
                raise ValueError(f"some keys<{keys}> not exists. {p.schema.names}")
            if p.party in psi_keys:
                raise ValueError(f"duplicate party<{p.party}>")
            csv_path = os.path.join(root_dir, p.uri)
            pyu = PYU(p.party)
            psi_keys[pyu] = keys
            psi_input_paths[pyu] = csv_path
            psi_output_paths[pyu] = output_csv_path

        spu = ctx.make_spu()
        with PathCleanUp({p.party: root_dir for p in inputs}):
            with ctx.trace_io():
                download_res = [
                    PYU(p.party)(download_csv)(
                        ctx.storage, p, psi_input_paths[PYU(p.party)], na_rep
                    )
                    for p in inputs
                ]
                wait(download_res)
            with ctx.trace_running():
                spu.psi_csv(
                    key=psi_keys,
                    input_path=psi_input_paths,
                    output_path=psi_output_paths,
                    receiver=receiver,
                    broadcast_result=True,
                    protocol="ECDH_PSI_3PC",
                    curve_type=self.ecdh_curve,
                )

            with ctx.trace_io():
                for pyu, path in psi_output_paths.items():
                    schema = input_parties[pyu.party].schema.to_arrow()
                    num_rows = pyu(upload_orc)(
                        ctx.storage, self.output_ds.uri, path, schema, na_rep
                    )
                num_rows = reveal(num_rows)

        output_tbl = VTable(
            self.output_ds.uri, input_parties, num_rows, self.input_ds1.system_info
        )
        self.output_ds.data = output_tbl.to_distdata()
        self.output_ds.data = output_tbl.to_distdata()
