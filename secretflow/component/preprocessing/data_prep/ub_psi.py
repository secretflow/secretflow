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
import logging
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
    VTable,
    VTableParty,
    VTableSchema,
    download_csv,
    register,
    upload_orc,
    uuid4,
)
from secretflow.component.preprocessing.data_prep.pis_utils import trans_keys_to_ids
from secretflow.component.preprocessing.data_prep.psi import JoinType, to_join_type
from secretflow.device import PYU, reveal, wait


@register(domain="data_prep", version="1.0.0")
class UnbalancePsi(Component):
    '''
    Unbalance psi with cache.
    '''

    join_type: JoinType = Field.union_attr(
        desc="join type, default is inner join.",
        default="inner_join",
    )
    allow_empty_result: bool = Field.attr(
        desc="Whether to allow the result to be empty, if allowed, an empty file will be saved, if not, an error will be reported.",
        default=False,
    )
    receiver_parties: list[str] = Field.party_attr(
        desc="Party names of receiver for result, all party will be receivers default; if only one party receive result, the result will be single-party table, hence you can not connect it to component with union table input.",
        list_limit=Interval.closed(0, 2),
    )
    keys: list[str] = Field.table_column_attr(
        'client_ds',
        desc="Keys to be used for psi.",
    )
    client_ds: Input = Field.input(
        desc="Client dataset.",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )
    cache: Input = Field.input(
        desc="Server cache.",
        types=[DistDataType.UNBALANCE_PSI_CACHE],
    )
    output_ds: Output = Field.output(
        desc="Output table",
        types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        client_tbl = VTable.from_distdata(self.client_ds).get_party(0)
        client_tbl.schema = trans_keys_to_ids(client_tbl.schema, self.keys)
        data_client = client_tbl.party
        random_str = uuid4(data_client)
        task_dir = os.path.join(ctx.data_dir, random_str)
        cache = ctx.load_tarfile(
            self.cache,
            DistDataType.UNBALANCE_PSI_CACHE,
            version=UB_PSI_CACHE_MAX,
            base_dir=task_dir,
        )
        public_info = cache.public_info
        server_party = public_info['server']
        client_party = public_info['client']
        cache_dir_name = public_info['cache_dir_name']
        na_rep = public_info['server_csv_na_rep']
        cache_path = os.path.join(task_dir, cache_dir_name)

        join_type = self.join_type.get_selected()

        with PathCleanUp({server_party: task_dir, client_party: task_dir}):
            client_csv_path = os.path.join(task_dir, f'{random_str}.client_ds.csv')

            if client_party != data_client:
                raise ValueError(
                    f"Client party in cache is mismatch with the input, {data_client} vs {client_party}."
                )
            # for consistency, use server null str, so server NULL and client NULL can be joined
            download_res = PYU(client_party)(download_csv)(
                ctx.storage, client_tbl, client_csv_path, na_rep
            )
            wait(download_res)

            server_get_result = True
            client_get_result = True
            receiver_parties = set(self.receiver_parties)
            for party in receiver_parties:
                if party not in [server_party, client_party]:
                    raise ValueError(
                        f"Receiver party {party} is not in [{server_party}, {client_party}]."
                    )
            if server_party not in receiver_parties:
                server_get_result = False
            if client_party not in receiver_parties:
                client_get_result = False

            output_path = {}
            if server_get_result:
                output_path[server_party] = os.path.join(
                    task_dir, f'{random_str}.server_output.csv'
                )
            if client_get_result:
                output_path[client_party] = os.path.join(
                    task_dir, f'{random_str}.client_output.csv'
                )

            spu = ctx.make_spu()
            with ctx.trace_running():
                spu.ub_psi(
                    mode="MODE_ONLINE",
                    role={server_party: "ROLE_SERVER", client_party: "ROLE_CLIENT"},
                    input_path={
                        client_party: client_csv_path,
                    },
                    keys={client_party: self.keys},
                    cache_path={
                        server_party: cache_path,
                        client_party: cache_path,
                    },
                    server_get_result=server_get_result,
                    client_get_result=client_get_result,
                    disable_alignment=not (server_get_result and client_get_result),
                    output_path=output_path,
                    left_side=(
                        self.join_type.left_join.left_side
                        if join_type == "left_join"
                        else ""
                    ),
                    join_type=to_join_type(join_type),
                    null_rep=na_rep,
                )
            parties = {}
            server_schema = VTableSchema.from_pb_str(
                base64.b64decode(public_info['server_csv_schema'].encode())
            )
            res = []
            if server_get_result:
                res.append(
                    PYU(server_party)(upload_orc)(
                        ctx.storage,
                        self.output_ds.uri,
                        output_path[server_party],
                        server_schema,
                        null_values=na_rep,
                    )
                )

                parties[server_party] = VTableParty(
                    party=server_party,
                    uri=self.output_ds.uri,
                    format='orc',
                    schema=server_schema,
                )
            if client_get_result:
                res.append(
                    PYU(client_party)(upload_orc)(
                        ctx.storage,
                        self.output_ds.uri,
                        output_path[client_party],
                        client_tbl.schema,
                        null_values=na_rep,
                    )
                )

                parties[client_party] = VTableParty(
                    party=client_party,
                    uri=self.output_ds.uri,
                    format='orc',
                    schema=client_tbl.schema,
                )
            res = set(reveal(res))
            if len(res) != 1:
                raise ValueError(f"psi result line count not match: {res}")
            logging.info(f"psi result line count: {res}")

            result_row_count = res.pop()
            if result_row_count == 0 and self.allow_empty_result == False:
                raise ValueError(f"psi result is emptys")

            result = VTable(
                name=self.output_ds.uri,
                parties=parties,
                system_info=self.client_ds.system_info,
                line_count=result_row_count,
            )

            self.output_ds.data = result.to_distdata()
