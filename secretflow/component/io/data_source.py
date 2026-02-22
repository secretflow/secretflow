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


import json
from urllib.parse import parse_qs, urlparse

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Output,
    Storage,
    VTable,
    VTableFieldKind,
    VTableFormat,
    VTableParty,
    new_connector,
    register,
)
from secretflow.device import PYU, reveal


@register(domain="io", version="1.0.0")
class DataSource(Component):
    """
    import data from an external data source
    """

    party: str = Field.party_attr(desc="")
    uri: str = Field.attr(
        desc=(
            "input uri, the uri format is "
            "datamesh:///{relative_path}?domaindata_id={domaindata_id}&datasource_id={datasource_id}&partition_spec={partition_spec}"
        ),
    )

    columns: str = Field.attr(
        desc='table column info, json format, for example {"col1": "ID", "col2":"FEATURE", "col3":"LABEL"}',
        default="",
    )

    output_ds: Output = Field.output(
        desc="output dataset",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        def download_table(
            storage: Storage,
            data_dir: str,
            input_uri: str,
            output_uri: str,
            party: str,
            columns_str: str,
        ) -> VTable:
            uri = urlparse(input_uri)
            if not uri.path:
                raise ValueError(f"invalid output path, {input_uri}")
            input_path = uri.path
            input_params = {k: values[0] for k, values in parse_qs(uri.query).items()}
            conn = new_connector(uri.scheme)
            info = conn.download_table(
                storage,
                data_dir,
                input_path,
                input_params,
                output_uri,
                VTableFormat.ORC,
            )

            if columns_str:
                columns = json.loads(columns_str)
                for f in info.schema:
                    if f.name not in columns:
                        continue
                    kind = VTableFieldKind.from_str(columns[f.name])
                    f.kind = kind

            parties = [
                VTableParty(party, output_uri, VTableFormat.ORC, schema=info.schema)
            ]
            tbl = VTable(output_uri, parties=parties, line_count=info.line_count)
            return tbl

        pyu = PYU(self.party)
        res = pyu(download_table)(
            ctx.storage,
            ctx.data_dir,
            self.uri,
            self.output_ds.uri,
            self.party,
            self.columns,
        )
        tbl: VTable = reveal(res)
        self.output_ds.data = tbl.to_distdata()
