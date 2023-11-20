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

import itertools
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from threading import Thread
from typing import Dict, List, Tuple

import jax
import pandas as pd

from secretflow.component.data_utils import DistDataType
from secretflow.component.entry import get_comp_def
from secretflow.spec.extend.cluster_pb2 import SFClusterDesc
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam, NodeEvalResult

from .node_controller import NodeController
from .test_case import (
    ClusterCase,
    DAGInput,
    DataCase,
    NetCase,
    PipelineCase,
    TestComp,
    TestNode,
    TestVersion,
)


@dataclass
class BenchmarkRecord:
    test_name: str
    mem_peak: float = -1
    run_time: float = -1
    # "wait / finished / failed"
    status: str = "wait"


class TestController(object):
    def __init__(self, aci_mode=False):
        self.sf_versions: List[TestVersion] = []
        self.nodes_controller: Dict[str, NodeController] = {}
        self.nodes: Dict[str, TestNode] = {}

        self.net_cases: List[NetCase] = []
        self.cluster_cases: List[ClusterCase] = []
        self.data_cases: List[DataCase] = []
        self.pipeline_cases: List[PipelineCase] = []

        self.aci_mode = aci_mode
        # in aci_mode, use fixed version / node / net / cluster / data case
        if aci_mode:
            self._init_aci_mode()

        self.record: Dict = {}

    def _init_aci_mode(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        ds = load_breast_cancer()
        x, y = scaler.fit_transform(ds["data"]), ds["target"]
        local_fs_paths = {}
        # local 3 parties
        for i, party in enumerate(["alice", "bob"]):
            local_fs_paths[party] = os.path.join(
                tempfile.gettempdir(), str(uuid.uuid4())
            )
            os.makedirs(local_fs_paths[party])
            os.makedirs(os.path.join(local_fs_paths[party], "logs"))
            node = TestNode(
                party=party,
                local_fs_path=local_fs_paths[party],
                rayfed_port=62310 + i * 5,
                spu_port=62311 + i * 5,
                hostname="127.0.0.1",
            )
            self.nodes[party] = node
            self.nodes_controller[party] = NodeController(node, True)
            data_path = os.path.join(local_fs_paths[party], "in.csv")
            ds = pd.DataFrame(
                x[:, i * 15 : (i + 1) * 15], columns=[f"{party}{c}" for c in range(15)]
            )
            if party == "alice":
                y = pd.DataFrame(y, columns=["y"])
                ds = pd.concat([ds, y], axis=1)
            ds.to_csv(data_path, index=True, index_label=f"id{i}")

        # use local python
        self.sf_versions.append(TestVersion("python", whl_paths=["local"]))
        # no net limit
        self.net_cases.append(NetCase("LAN", 0, 0))
        # only test SEMI2K-2PC
        self.cluster_cases.append(
            ClusterCase("SEMI2K", "SEMI2K", "FM64", 18, ["alice", "bob"])
        )
        # only test breast_cancer data
        alice = DAGInput(
            DistDataType.INDIVIDUAL_TABLE,
            {"alice": "in.csv"},
            feature_columns={"alice": [f"alice{c}" for c in range(15)]},
            feature_types={"alice": ["float32"] * 15},
            id_columns={"alice": ["id0"]},
            id_types={"alice": ["str"]},
            label_columns={"alice": ["y"]},
            label_types={"alice": ["float32"]},
        )
        bob = DAGInput(
            DistDataType.INDIVIDUAL_TABLE,
            {"bob": "in.csv"},
            feature_columns={"bob": [f"bob{c}" for c in range(15)]},
            feature_types={"bob": ["float32"] * 15},
            id_columns={"bob": ["id1"]},
            id_types={"bob": ["str"]},
        )
        data_case = DataCase("breast_cancer", {"alice": alice, "bob": bob})
        self.data_cases.append(data_case)

    # define test version
    def add_test_version(self, ver: TestVersion) -> None:
        assert not self.aci_mode
        self.sf_versions.append(ver)

    # where to test
    def add_node(self, node: TestNode) -> None:
        assert not self.aci_mode
        self.nodes_controller[node.party] = NodeController(node)
        self.nodes[node.party] = node

    # test cases
    def add_net_case(self, net: NetCase) -> None:
        assert not self.aci_mode
        self.net_cases.append(net)

    def add_cluster_case(self, clu: ClusterCase) -> None:
        assert not self.aci_mode
        self.cluster_cases.append(clu)

    def add_data_case(self, data: DataCase) -> None:
        assert not self.aci_mode
        self.data_cases.append(data)

    def add_pipeline_case(self, pipe: PipelineCase) -> None:
        self.pipeline_cases.append(pipe)

    # run tests
    def _init_env(self) -> None:
        if not hasattr(self, "current_ver") or self.current_ver != self.ver_case:
            logging.info(f"start test for version {self.ver_case}")
            for name in self.nodes_controller:
                logging.info(f"init env for node {name}")
                self.nodes_controller[name].init_env(self.ver_case)
            self.current_ver = self.ver_case

    def _init_net(self) -> None:
        if not hasattr(self, "current_net") or self.current_net != self.net_case:
            logging.info(f"start test for net limit {self.net_case.name}")
            for name in self.nodes_controller:
                logging.info(f"init net for node {name}")
                self.nodes_controller[name].limit_network(self.net_case)
            self.current_net = self.net_case

    def _run_comp(
        self, comp: TestComp, eval: NodeEvalParam, cluster: ClusterCase
    ) -> Tuple[float, float, NodeEvalResult]:
        comp_name = (
            f"{comp.uid}:{comp.comp_domain}:{comp.comp_name}:{comp.comp_version}"
        )
        logging.info(f"run comp {comp_name}")
        results = [None] * len(self.nodes_controller)
        work_threads = []
        sf_cluster_desc = self._build_sf_cluster(cluster)
        for idx, node in enumerate(self.nodes_controller.values()):

            def work_thread(i):
                try:
                    result = node.run_comp(comp.uid, eval, sf_cluster_desc, self.nodes)
                    results[i] = result
                except Exception as e:
                    results[i] = e

            t = Thread(target=work_thread, args=(idx,))
            t.start()
            work_threads.append(t)

        while True:
            all_stoped = True
            for i in range(len(results)):
                if work_threads[i].is_alive():
                    all_stoped = False
                else:
                    if isinstance(results[i], Exception):
                        logging.error(f"run comp {comp} failed, error {results[i]}")
                        raise results[i]

            if all_stoped:
                break

            time.sleep(0.1)

        for r in results:
            if isinstance(r, Exception):
                logging.error(f"run comp {comp} failed, error {r}")
                raise r

        results = list(zip(*results))

        logging.info(
            f"run comp {comp_name} finished, mem peak {max(results[0])}, time {max(results[1])}"
        )

        return max(results[0]), max(results[1]), results[2][0]

    def _build_sf_cluster(self, clu: ClusterCase) -> SFClusterDesc:
        ret = SFClusterDesc()
        ret.sf_version = str(
            self.ver_case.version
            if self.ver_case.source == "public"
            else self.ver_case.whl_paths
        )
        ret.py_version = "3.8"
        ret.parties.extend(self.nodes.keys())

        spu = SFClusterDesc.DeviceDesc()
        spu.name = "spu"
        spu.type = "spu"
        spu.parties.extend(clu.spu_parties)
        spu.config = json.dumps(
            {
                "runtime_config": {
                    "protocol": clu.protocol,
                    "field": clu.field,
                    "fxp_fraction_bits": clu.fxp,
                },
                "link_desc": {
                    "connect_retry_times": 20,
                    "connect_retry_interval_ms": 1000,
                    "brpc_channel_protocol": "http",
                    "brpc_channel_connection_type": "pooled",
                    "recv_timeout_ms": 3600 * 1000,
                    "http_timeout_ms": 600 * 1000,
                },
            }
        )
        ret.devices.append(spu)

        heu = SFClusterDesc.DeviceDesc(
            name="heu",
            type="heu",
            parties=[],
            config=json.dumps(
                {
                    "mode": "PHEU",
                    "schema": "ou",
                    "key_size": 2048,
                }
            ),
        )
        ret.devices.append(heu)

        return ret

    def _build_table_schema(self, data: DAGInput, party: str) -> TableSchema:
        ret = TableSchema()
        if data.id_columns:
            assert party in data.id_columns
            ret.ids.extend(data.id_columns[party])
            ret.id_types.extend(data.id_types[party])

        assert data.feature_columns is not None
        assert party in data.feature_columns
        ret.features.extend(data.feature_columns[party])

        assert data.feature_types is not None
        assert party in data.feature_types
        ret.feature_types.extend(data.feature_types[party])

        assert len(ret.feature_types) == len(ret.features)

        if data.label_columns and party in data.label_columns:
            ret.labels.extend(data.label_columns[party])
            ret.label_types.extend(data.label_types[party])
        return ret

    def _build_root_dist_data(self, name: str, sf_clus: ClusterCase) -> DistData:
        data = self.data_case.get_data(name)

        ret = DistData()
        ret.name = name
        ret.type = str(data.data_type)
        ret.system_info.app = "Secretflow"
        ret.system_info.app_meta.Pack(self._build_sf_cluster(sf_clus))

        if data.data_type == DistDataType.INDIVIDUAL_TABLE:
            assert len(data.data_paths) == 1
            meta = IndividualTable()
            for party in data.data_paths:
                meta.schema.CopyFrom(self._build_table_schema(data, party))
            meta.line_count = -1
        else:
            meta = VerticalTable()
            for party in data.data_paths:
                meta.schemas.append(self._build_table_schema(data, party))
            meta.line_count = -1
        ret.meta.Pack(meta)

        for party in data.data_paths:
            data_ref = DistData.DataRef()
            data_ref.uri = data.data_paths[party]
            data_ref.party = party
            # only csv for now
            data_ref.format = "csv"
            ret.data_refs.append(data_ref)

        return ret

    def _build_eval_param(
        self, comp: TestComp, inputs: List[DistData], pipe_name: str
    ) -> NodeEvalParam:
        ret = NodeEvalParam()
        ret.domain = comp.comp_domain
        ret.name = comp.comp_name
        ret.version = comp.comp_version
        for key in comp.attrs:
            ret.attr_paths.append(key)
            atomic_param = Attribute()
            value = comp.attrs[key]
            # note instance(True, int) will be evaluated to True.
            if isinstance(value, bool):
                atomic_param.b = value
            elif isinstance(value, int):
                atomic_param.i64 = value
            elif isinstance(value, float):
                atomic_param.f = value
            elif isinstance(value, str):
                atomic_param.s = value
            elif isinstance(value, list):
                if len(value):
                    if isinstance(value[0], bool):
                        atomic_param.bs.extend(value)
                    elif isinstance(value[0], int):
                        atomic_param.i64s.extend(value)
                    elif isinstance(value[0], float):
                        atomic_param.fs.extend(value)
                    elif isinstance(value[0], str):
                        atomic_param.ss.extend(value)
            else:
                raise RuntimeError("not supported type")
            ret.attrs.append(atomic_param)
        ret.inputs.extend(inputs)
        comp_def = get_comp_def(comp.comp_domain, comp.comp_name, comp.comp_version)
        outputs = len(comp_def.outputs)
        ret.output_uris.extend([str(uuid.uuid4()) for _ in range(outputs)])
        return ret

    def _init_pipeline_record(self) -> None:
        sf_version = str(
            self.ver_case.version
            if self.ver_case.source == "public"
            else self.ver_case.whl_paths
        )
        if sf_version not in self.record:
            self.record[sf_version] = {}
        ver_record = self.record[sf_version]

        if self.net_case.name not in ver_record:
            ver_record[self.net_case.name] = {}
        net_record = ver_record[self.net_case.name]

        if self.cluster_case.name not in net_record:
            net_record[self.cluster_case.name] = {}
        clu_record = net_record[self.cluster_case.name]

        if self.data_case.name not in clu_record:
            clu_record[self.data_case.name] = {}
        data_record = clu_record[self.data_case.name]

        if self.pipeline_case.name not in data_record:
            data_record[self.pipeline_case.name] = {}
        pipe_record = data_record[self.pipeline_case.name]

        for comp in self.pipeline_case.comp_inputs:
            pipe_record[comp] = BenchmarkRecord(comp)
        self.pipe_record = pipe_record

    def _record_comp(
        self, comp: str, failed: bool, mem: float = -1, run_time: float = -1
    ) -> None:
        assert comp in self.pipe_record
        self.pipe_record[comp].status = "finished" if not failed else "failed"
        self.pipe_record[comp].mem_peak = mem
        self.pipe_record[comp].run_time = run_time

    def _run_pipeline(self) -> None:
        cluster = self.cluster_case
        pipe = self.pipeline_case
        logging.info(
            f"start test for pipeline {pipe.name} "
            f"on dataset {self.data_case.name} "
            f"with cluster config {cluster.name}"
        )

        comp_outputs = dict()  # [str, List[DistData]]
        waiting_comps = pipe.comp_inputs.copy()
        self._init_pipeline_record()

        def get_comp_inputs(inputs: List[str]) -> List[DistData]:
            ret = []
            for i in inputs:
                in_parts = i.split(".")
                assert len(in_parts) == 2
                upstream_comp, output = in_parts
                if upstream_comp == "DAGInput":
                    ret.append(self._build_root_dist_data(output, cluster))
                elif upstream_comp in comp_outputs:
                    ret.append(comp_outputs[upstream_comp][int(output)])
                else:
                    return None
            return ret

        while len(waiting_comps):
            has_ready_comp = False
            for n in waiting_comps:
                inputs = get_comp_inputs(waiting_comps[n])
                if inputs is not None:
                    comp = pipe.comps[n]
                    eval_param = self._build_eval_param(comp, inputs, pipe.name)
                    try:
                        mem, time, outputs = self._run_comp(comp, eval_param, cluster)
                        self._record_comp(n, False, mem, time)
                    except Exception as e:
                        self._record_comp(n, True)
                        raise e from None
                    comp_outputs[n] = list(outputs.outputs)
                    waiting_comps.pop(n)
                    has_ready_comp = True
                    break

            assert has_ready_comp

    def run(self, stop_on_fail_case: bool = False) -> Dict:
        for test_case in itertools.product(
            self.sf_versions,
            self.net_cases,
            self.cluster_cases,
            self.data_cases,
            self.pipeline_cases,
        ):
            (
                self.ver_case,
                self.net_case,
                self.cluster_case,
                self.data_case,
                self.pipeline_case,
            ) = test_case

            self._init_env()
            self._init_net()
            try:
                self._run_pipeline()
            except Exception as e:
                logging.exception(f"error while running pipe {self.pipeline_case.name}")
                if stop_on_fail_case:
                    raise e from None

        for n in self.nodes_controller:
            try:
                self.nodes_controller[n].clear_up()
            except Exception as e:
                logging.error(f"error while clear_up : {e}")

        flatten_value, tree = jax.tree_util.tree_flatten(self.record)
        flatten_value = [asdict(v) if is_dataclass(v) else v for v in flatten_value]
        return jax.tree_util.tree_unflatten(tree, flatten_value)
