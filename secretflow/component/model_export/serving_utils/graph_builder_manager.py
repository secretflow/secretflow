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

import io
from typing import Dict, List, Union

import pyarrow as pa
from secretflow_serving_lib import GraphBuilder, build_serving_tar

from secretflow.component.storage import ComponentStorage
from secretflow.device import PYU, PYUObject, proxy, wait
from secretflow.device.device.heu import HEU


class GraphBuilderManager:
    def __init__(self, pyus: List[PYU]) -> None:
        pyu_builder_actor = proxy(PYUObject)(GraphBuilder)
        self.graph_builders = {p: pyu_builder_actor(device=p) for p in pyus}
        self.pyus = pyus
        self.node_names = []

    def add_node(
        self,
        node_name: str,
        op: str,
        input_schemas: Dict[PYU, Union[pa.Schema, List[pa.Schema]]],
        output_schemas: Dict[PYU, pa.Schema],
        party_kwargs: Dict[PYU, Dict],
        parents: List[str] = None,
    ):
        assert set(party_kwargs) == set(self.graph_builders)
        assert set(input_schemas) == set(self.graph_builders)
        assert set(output_schemas) == set(self.graph_builders)
        waits = []
        for pyu in party_kwargs:
            builder = self.graph_builders[pyu]
            kwargs = party_kwargs[pyu]
            node_parnents = parents
            if node_parnents is None:
                node_parnents = [self.node_names[-1]] if self.node_names else []
            waits.append(
                builder.add_node(
                    name=node_name, parents_name=node_parnents, op=op, **kwargs
                )
            )
        wait(waits)
        self.node_names.append(node_name)

    def get_last_node_name(self):
        if len(self.node_names):
            return self.node_names[-1]
        else:
            return None

    def new_execution(
        self,
        dp_type: str,
        session_run: bool = False,
        party_specific_flag: Dict[PYU, bool] = None,
    ):
        assert party_specific_flag is None or set(party_specific_flag) == set(
            self.graph_builders
        )
        waits = []
        for pyu in self.graph_builders:
            builder = self.graph_builders[pyu]
            specific_flag = party_specific_flag[pyu] if party_specific_flag else False
            waits.append(
                builder.begin_new_execution(
                    dispatch_type=dp_type,
                    session_run=session_run,
                    specific_flag=specific_flag,
                )
            )
        wait(waits)

    def set_he_config(self, heu_dict: Dict[str, HEU]):
        def _key_serialize(key):
            return key.serialize()

        waits = []
        for pyu in self.graph_builders:
            heu = heu_dict[pyu.__str__()]
            builder = self.graph_builders[pyu]
            pk = heu.get_participant(pyu.__str__()).public_key.remote()
            sk = heu.get_participant(pyu.__str__()).secret_key.remote()

            pk_bytes = pyu(_key_serialize)(pk)
            sk_bytes = pyu(_key_serialize)(sk)

            waits.append(builder.set_he_config(pk_bytes, sk_bytes, heu.scale))
        wait(waits)

    def dump_tar_files(self, name, desc, ctx, uri) -> None:
        waits = []

        def dump_io(io_handle: io.BytesIO, comp_storage: ComponentStorage, uri: str):
            with comp_storage.get_writer(uri) as f:
                f.write(io_handle.getvalue())

        for pyu, builder in self.graph_builders.items():
            proto = builder.build_proto()
            io_handle = pyu(build_serving_tar)(name, desc, proto)
            waits.append(pyu(dump_io)(io_handle, ctx.comp_storage, uri))

        wait(waits)
