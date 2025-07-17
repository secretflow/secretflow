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


import base64
import os

from secretflow.component.core import (
    UB_PSI_CACHE_MAX,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    PathCleanUp,
    TarFile,
    VTable,
    download_csv,
    register,
    uuid4,
)
from secretflow.component.preprocessing.data_prep.pis_utils import trans_keys_to_ids
from secretflow.device import PYU, wait


@register(domain="data_prep", version="1.0.0")
class UnbalancePsiCache(Component):
    '''
    Generate cache for unbalance psi on both sides.
    '''

    client: str = Field.party_attr(
        desc="Party of client(party with the smaller dataset)."
    )
    keys: list[str] = Field.table_column_attr(
        'input_ds',
        desc="Keys to be used for psi.",
        limit=Interval.closed_open(1, None),
    )
    input_ds: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )
    output_cache: Output = Field.output(
        desc="Output cache.",
        types=[DistDataType.UNBALANCE_PSI_CACHE],
    )

    def evaluate(self, ctx: Context):
        input_tbl = VTable.from_distdata(self.input_ds).get_party(0)
        server_party = input_tbl.party
        if server_party == self.client:
            raise ValueError(
                f"server party {server_party} should not be the same as client party {self.client}"
            )

        input_tbl.schema = trans_keys_to_ids(input_tbl.schema, self.keys)

        schema = input_tbl.schema.to_pb()
        server_pyu = PYU(server_party)
        client_pyu = PYU(self.client)
        for key in self.keys:
            if key not in input_tbl.columns:
                raise ValueError(
                    f"keys {key} not in table columns {input_tbl.columns}."
                )

        random_str = uuid4(server_party)
        na_rep = random_str

        root_dir = os.path.join(ctx.data_dir, random_str)
        server_csv_path = os.path.join(root_dir, f'{random_str}.csv')
        cache_dir_name = 'cache'
        cache_path = os.path.join(root_dir, cache_dir_name)

        with PathCleanUp({server_party: root_dir, self.client: root_dir}):
            with ctx.trace_io():
                wait(
                    [
                        server_pyu(download_csv)(
                            ctx.storage, input_tbl, server_csv_path, na_rep
                        )
                    ]
                )
            spu = ctx.make_spu()
            with ctx.trace_running():
                spu.ub_psi(
                    mode="MODE_OFFLINE",
                    role={server_party: "ROLE_SERVER", self.client: "ROLE_CLIENT"},
                    input_path={server_party: server_csv_path},
                    keys={server_party: self.keys},
                    cache_path={server_party: cache_path, self.client: cache_path},
                )

            cache = TarFile(
                name=self.output_cache.uri,
                type=DistDataType.UNBALANCE_PSI_CACHE,
                version=UB_PSI_CACHE_MAX,
                files={server_pyu.party: [cache_path], client_pyu.party: [cache_path]},
                public_info={
                    'server_csv_schema': base64.b64encode(
                        schema.SerializeToString()
                    ).decode(),
                    'server_csv_na_rep': na_rep,
                    'cache_dir_name': cache_dir_name,
                    "server": server_party,
                    "client": self.client,
                },
                system_info=self.input_ds.system_info,
            )
            ctx.dump_to(cache, self.output_cache)
