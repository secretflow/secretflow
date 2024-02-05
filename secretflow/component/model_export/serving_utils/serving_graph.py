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


from __future__ import annotations

import io
import tarfile
from typing import Any, List

import numpy as np
import secretflow_serving_lib as sfs
from google.protobuf import json_format
from secretflow_serving_lib.graph_pb2 import DispatchType


class _Execution:
    def __init__(self, dis_type: str, session_run: bool, specific_flag: bool):
        # all exec nodes in current execution phase
        self.node_list: List[sfs.graph_pb2.NodeDef] = list()
        self.last_node_is_returnable = False
        # rt conf for this execution phase
        self.rt_conf = sfs.graph_pb2.RuntimeConfig(
            dispatch_type=DispatchType.Value(dis_type),
            session_run=session_run,
            specific_flag=specific_flag,
        )


def __init_ops():
    def op_wrapper(op_def: sfs.op_pb2.OpDef):
        def get_op_attr(
            type: sfs.attr_pb2.AttrType, param: Any, name
        ) -> sfs.attr_pb2.AttrValue:
            ret = sfs.attr_pb2.AttrValue()
            if type == sfs.attr_pb2.AT_BOOL:
                ret.b = bool(param)
            elif type == sfs.attr_pb2.AT_FLOAT:
                ret.f = np.float32(param)
            elif type == sfs.attr_pb2.AT_DOUBLE:
                ret.d = np.float64(param)
            elif type == sfs.attr_pb2.AT_INT32:
                ret.i32 = np.int32(param)
            elif type == sfs.attr_pb2.AT_INT64:
                ret.i64 = np.int64(param)
            elif type == sfs.attr_pb2.AT_STRING:
                assert isinstance(param, (np.str_, str))
                ret.s = param
            elif type == sfs.attr_pb2.AT_BYTES:
                assert isinstance(param, (np.bytes_, bytes))
                ret.by = param

            elif type == sfs.attr_pb2.AT_BOOL_LIST:
                ret.bs.data.extend(map(bool, param))
            elif type == sfs.attr_pb2.AT_INT32_LIST:
                ret.i32s.data.extend(map(np.int32, param))
            elif type == sfs.attr_pb2.AT_INT64_LIST:
                ret.i64s.data.extend(map(np.int64, param))
            elif type == sfs.attr_pb2.AT_FLOAT_LIST:
                ret.fs.data.extend(map(np.float32, param))
            elif type == sfs.attr_pb2.AT_DOUBLE_LIST:
                ret.ds.data.extend(map(np.float64, param))
            elif type == sfs.attr_pb2.AT_STRING_LIST:
                assert isinstance(param, list) and all(
                    [isinstance(p, (np.str_, str)) for p in param]
                )
                ret.ss.data.extend(param)
            elif type == sfs.attr_pb2.AT_BYTES_LIST:
                assert isinstance(param, list) and all(
                    [isinstance(p, (np.bytes_, bytes)) for p in param]
                )
                ret.bys.data.extend(param)

            return ret

        def op_fun(self: _Execution, unique_node_name, node_parents, **kwargs):
            # check if kwargs match to op_def
            input_keys = set(kwargs)
            op_kwargs = dict()
            for attr in op_def.attrs:
                if attr.name not in kwargs:
                    assert attr.is_optional, f"missing necessary attr {attr.name}"
                else:
                    input_keys.remove(attr.name)

                    try:
                        op_kwargs[attr.name] = get_op_attr(
                            attr.type, kwargs[attr.name], attr.name
                        )
                    except Exception as e:
                        raise AssertionError(
                            f"get_op_attr err {e} on attr.name {attr.name}, "
                            f"attr.type {sfs.attr_pb2.AttrType.Name(attr.type)}, "
                            f"attr {kwargs[attr.name]}, attr type {type(kwargs[attr.name])}"
                        )
            assert len(input_keys) == 0, f"unknown attr {input_keys}"

            if (
                self.rt_conf.dispatch_type != DispatchType.DP_ALL
                and len(self.node_list) == 0
            ):
                assert (
                    op_def.tag.mergeable
                ), f"first op in Execution need be mergeable if exec is not DP_ALL type"

            if not op_def.tag.variable_inputs:
                assert (
                    len(node_parents) == len(op_def.inputs) or len(node_parents) == 0
                ), f"num of node({unique_node_name}) parents not match the op({unique_node_name})'s num of inputs, {len(node_parents)} vs {len(op_def.inputs)}"

            exec_node = sfs.graph_pb2.NodeDef()
            exec_node.name = unique_node_name
            exec_node.op = op_def.name
            exec_node.parents.extend(node_parents)

            exec_node.op_version = op_def.version
            for k, v in op_kwargs.items():
                exec_node.attr_values[k].CopyFrom(v)

            self.node_list.append(exec_node)
            self.last_node_is_returnable = op_def.tag.returnable

            return self

        op_fun.__name__ = op_def.name
        op_fun.__qualname__ = op_def.name
        op_fun.__doc__ = op_def.desc
        return op_fun

    all_op: List[sfs.op_pb2.OpDef] = sfs.api.get_all_ops()
    for op in all_op:
        setattr(_Execution, op.name.lower(), op_wrapper(op))


__init_ops()


class _Graph:
    def __init__(self):
        self.executions: List[_Execution] = []

    def add(self, exec_phase: _Execution):
        assert exec_phase not in self.executions
        assert exec_phase.node_list
        self.executions.append(exec_phase)

    def save(self) -> sfs.graph_pb2.GraphDef:
        node_names = set()
        ret = sfs.graph_pb2.GraphDef()
        ret.version = sfs.get_graph_version()
        assert self.executions
        for execution in self.executions:
            exec_def = sfs.graph_pb2.ExecutionDef()
            exec_def.config.CopyFrom(execution.rt_conf)
            for node in execution.node_list:
                assert node.name not in node_names
                node_names.add(node.name)
                ret.node_list.append(node)
                exec_def.nodes.append(node.name)
            ret.execution_list.append(exec_def)
        return ret


class GraphBuilder:
    def __init__(self) -> None:
        self.executions = []
        # first exec must be DP_ALL
        self.executions.append(_Execution("DP_ALL", False, False))

    def add_node(
        self, node_name: str, node_parents: List[str], op: str, **kwargs
    ) -> None:
        e_op = getattr(self.executions[-1], op, None)
        assert e_op, f"not exist op {op}"
        e_op(node_name, node_parents, **kwargs)

    def new_execution(self, dp_type: str, session_run: bool, specific_flag: bool):
        assert not session_run, "not session_run for now"
        self.executions.append(_Execution(dp_type, session_run, specific_flag))

    def dump_serving_tar(self, name: str, desc: str, uri: str) -> None:
        graph = _Graph()

        for i, exec in enumerate(self.executions):
            graph.add(exec)

        mb = sfs.bundle_pb2.ModelBundle()
        mb.name = name
        mb.desc = desc
        mb.graph.CopyFrom(graph.save())
        mb_data = mb.SerializeToString()

        mm = sfs.bundle_pb2.ModelManifest()
        mm.bundle_path = "model_file"
        mm.bundle_format = sfs.bundle_pb2.FF_PB
        mm_data = json_format.MessageToJson(mm, indent=0).encode("utf-8")

        fh = io.BytesIO()
        with tarfile.open(fileobj=fh, mode="w:gz") as tar:
            info = tarfile.TarInfo("MANIFEST")
            info.size = len(mm_data)
            info.mode = int('0666', base=8)
            tar.addfile(info, io.BytesIO(initial_bytes=mm_data))

            info = tarfile.TarInfo("model_file")
            info.size = len(mb_data)
            info.mode = int('0666', base=8)
            tar.addfile(info, io.BytesIO(initial_bytes=mb_data))

        with open(uri, "wb") as f:
            f.write(fh.getvalue())
