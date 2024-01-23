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
from typing import Dict, List

from secretflow.device import PYU, PYUObject, proxy, wait

from .serving_graph import GraphBuilder


class GraphBuilderManager:
    def __init__(self, pyus: List[PYU]) -> None:
        pyu_builder_actor = proxy(PYUObject)(GraphBuilder)
        self.graph_builders = {p: pyu_builder_actor(device=p) for p in pyus}
        self.pyus = pyus

    def add_node(self, node_name: str, op: str, party_kwargs: Dict[PYU, Dict]):
        assert set(party_kwargs) == set(self.graph_builders)
        waits = []
        for pyu in party_kwargs:
            builder = self.graph_builders[pyu]
            kwargs = party_kwargs[pyu]
            waits.append(builder.add_node(node_name, op, **kwargs))
        wait(waits)

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
            waits.append(builder.new_execution(dp_type, session_run, specific_flag))
        wait(waits)

    def dump_tar_files(self, name, desc, ctx, uri) -> None:
        # TODO: only local fs is supported at this moment.
        storage_root = ctx.local_fs_wd
        uri = os.path.join(storage_root, uri)
        waits = []
        for b in self.graph_builders.values():
            waits.append(b.dump_serving_tar(name, desc, uri))
        wait(waits)
