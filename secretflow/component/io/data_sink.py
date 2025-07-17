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


from urllib.parse import parse_qs, urlparse

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Storage,
    VTable,
    VTableParty,
    new_connector,
    register,
)
from secretflow.device import PYU, wait


@register(domain="io", version="1.0.0")
class DataSink(Component):
    '''
    export data to an external data source
    '''

    output_party: str = Field.party_attr(
        desc="output party", list_limit=Interval.closed(0, 1)
    )
    output_uri: str = Field.attr(
        desc=(
            "output uri, the uri format is "
            "datamesh:///{relative_path}?domaindata_id={domaindata_id}&datasource_id={datasource_id}&partition_spec={partition_spec}"
        )
    )

    input_data: Input = Field.input(
        desc="Input dist data",
        types=[
            DistDataType.INDIVIDUAL_TABLE,
            DistDataType.VERTICAL_TABLE,
        ],
    )

    def evaluate(self, ctx: Context):
        def upload_table(
            storage: Storage, data_dir: str, input_tbl: VTableParty, output_uri: str
        ):
            uri = urlparse(output_uri)
            if not uri.path:
                raise ValueError(f"invalid output path, {output_uri}")

            output_path = uri.path
            output_params = {k: values[0] for k, values in parse_qs(uri.query).items()}
            conn = new_connector(uri.scheme)
            conn.upload_table(
                storage,
                data_dir,
                input_tbl.uri,
                input_tbl.format,
                input_tbl.schema,
                output_path,
                output_params,
            )

        input_tbl = VTable.from_distdata(self.input_data)

        if not self.output_party:
            if len(input_tbl.parties) > 1:
                raise ValueError(f"empty party")
            self.output_party = input_tbl.get_party(0).party
        pyu = PYU(self.output_party)
        input_party = input_tbl.get_party(self.output_party)
        res = pyu(upload_table)(ctx.storage, ctx.data_dir, input_party, self.output_uri)
        wait(res)
