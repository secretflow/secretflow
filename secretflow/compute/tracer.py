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

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from secretflow.spec.v1 import compute_trace_pb2


class _TracerType(Enum):
    TABLE = 1
    ARROW = 2


class _TraceRunner:
    def __init__(self, dag: List[_Tracer]) -> None:
        assert dag[0].output_type is _TracerType.TABLE
        assert dag[0].inputs is None
        assert dag[-1].output_type is _TracerType.TABLE
        assert len(set(dag)) == len(dag)
        assert len(dag) > 1
        self.dag: List[_Tracer] = dag
        self.ref_count = defaultdict(int)
        for t in self.dag:
            if t.inputs is None:
                continue
            for i in t.inputs:
                if isinstance(i, _Tracer):
                    self.ref_count[i] += 1
        assert len(self.ref_count) > 0
        self.input_features = dag[0].output_schema.names
        if not isinstance(self.input_features, list):
            assert isinstance(self.input_features, str)
            self.input_features = [self.input_features]

    def _get_input(self, t: _Tracer):
        assert t in self.run_ref_count
        assert t in self.output_map
        assert self.run_ref_count[t] > 0
        self.run_ref_count[t] -= 1
        ret = self.output_map[t]
        if self.run_ref_count[t] == 0:
            del self.output_map[t]
        return ret

    def get_input_features(self):
        return self.input_features

    def dump_serving_pb(
        self, name: str
    ) -> Tuple[compute_trace_pb2.ComputeTrace, pa.Schema, pa.Schema]:
        return self.dag[-1].dump_serving_pb(name)

    def run(self, input: Union[pd.DataFrame, pa.Table]) -> pd.DataFrame:
        assert isinstance(input, (pd.DataFrame, pa.Table))
        if isinstance(input, pd.DataFrame):
            input = pa.Table.from_pandas(input)

        assert input.schema == self.dag[0].output_schema

        self.run_ref_count = self.ref_count.copy()
        self.output_map = dict()
        self.output_map[self.dag[0]] = input

        for t in self.dag[1:]:
            assert t not in self.output_map

            if t.operate == compute_trace_pb2.ExtendFunctionName.Name(
                compute_trace_pb2.EFN_TB_COLUMN
            ):
                assert len(t.inputs) == 2
                assert isinstance(t.inputs[0], _Tracer)
                input_table = self._get_input(t.inputs[0])
                assert isinstance(input_table, pa.Table)
                self.output_map[t] = input_table.column(t.inputs[1])
            elif t.operate == compute_trace_pb2.ExtendFunctionName.Name(
                compute_trace_pb2.EFN_TB_ADD_COLUMN
            ):
                assert len(t.inputs) == 4
                assert isinstance(t.inputs[0], _Tracer)
                assert isinstance(t.inputs[3], _Tracer)
                input_table = self._get_input(t.inputs[0])
                input_array = self._get_input(t.inputs[3])
                assert isinstance(input_table, pa.Table)
                assert isinstance(input_array, pa.ChunkedArray)
                self.output_map[t] = input_table.add_column(
                    t.inputs[1], t.inputs[2], input_array
                )
            elif t.operate == compute_trace_pb2.ExtendFunctionName.Name(
                compute_trace_pb2.EFN_TB_REMOVE_COLUMN
            ):
                assert len(t.inputs) == 2
                assert isinstance(t.inputs[0], _Tracer)
                input_table = self._get_input(t.inputs[0])
                assert isinstance(input_table, pa.Table)
                self.output_map[t] = input_table.remove_column(t.inputs[1])
            elif t.operate == compute_trace_pb2.ExtendFunctionName.Name(
                compute_trace_pb2.EFN_TB_SET_COLUMN
            ):
                assert len(t.inputs) == 4
                assert isinstance(t.inputs[0], _Tracer)
                assert isinstance(t.inputs[3], _Tracer)
                input_table = self._get_input(t.inputs[0])
                input_array = self._get_input(t.inputs[3])
                assert isinstance(input_table, pa.Table)
                assert isinstance(input_array, pa.ChunkedArray)
                self.output_map[t] = input_table.set_column(
                    t.inputs[1], t.inputs[2], input_array
                )
            else:
                py_func = getattr(pc, t.operate)
                py_inputs = []
                for i in t.inputs:
                    if isinstance(i, _Tracer):
                        array = self._get_input(i)
                        assert isinstance(array, pa.ChunkedArray), f"got {type(array)}"
                        py_inputs.append(array)
                    else:
                        py_inputs.append(i)
                py_inputs.extend(t.py_args)
                array = py_func(*py_inputs, **t.py_kwargs)
                self.output_map[t] = array

        assert len(self.output_map) == 1
        assert self.dag[-1] in self.output_map
        ret_table = self.output_map.pop(self.dag[-1])
        assert isinstance(ret_table, pa.Table)
        assert ret_table.schema == self.dag[-1].output_schema

        return ret_table.to_pandas()


class _Tracer:
    def __init__(
        self,
        operate: str,
        output_type: _TracerType,
        inputs: List[Union[_Tracer, int, float, str]] = None,
        py_args: List = None,
        py_kwargs: Dict = None,
        options: bytes = None,
        output_schema: pa.Schema = None,
    ):
        self.operate = operate
        self.output_type = output_type
        self.inputs = inputs
        self.py_args = py_args
        self.py_kwargs = py_kwargs
        self.options = options
        self.output_schema = output_schema

    def _flatten_dag(self) -> List[_Tracer]:
        dag_tracers: List[_Tracer] = []

        backtrace = [self]
        while len(backtrace):
            t = backtrace.pop(0)
            dag_tracers.append(t)
            if t.inputs is None:
                continue
            for i in t.inputs:
                if isinstance(i, _Tracer):
                    backtrace.append(i)

        assert len(dag_tracers) > 1
        dag_tracers.reverse()
        dag_tracers = list(dict.fromkeys(dag_tracers))
        assert dag_tracers[0].output_type is _TracerType.TABLE
        assert dag_tracers[0].inputs is None
        assert all(
            [d.inputs is not None for d in dag_tracers[1:]]
        ), "only allow one input in dag"
        assert dag_tracers[-1] == self
        assert dag_tracers[-1].output_type is _TracerType.TABLE

        return dag_tracers

    def dump_runner(self) -> _TraceRunner:
        return _TraceRunner(self._flatten_dag())

    def column_changes(self) -> Tuple[List, List]:
        dag = self._flatten_dag()
        remove_cols = {}
        append_cols = {}

        def _get_name(t: _Tracer, i: int) -> str:
            return t.inputs[0].output_schema.field(i).name

        def _append(n: str):
            assert n not in append_cols
            append_cols[n] = None

        def _remove(n: str):
            if n in append_cols:
                del append_cols[n]
            else:
                remove_cols[n] = None

        for t in dag[1:]:
            if t.operate == compute_trace_pb2.ExtendFunctionName.Name(
                compute_trace_pb2.EFN_TB_ADD_COLUMN
            ):
                _append(t.inputs[2])
            elif t.operate == compute_trace_pb2.ExtendFunctionName.Name(
                compute_trace_pb2.EFN_TB_REMOVE_COLUMN
            ):
                _remove(_get_name(t, t.inputs[1]))
            elif t.operate == compute_trace_pb2.ExtendFunctionName.Name(
                compute_trace_pb2.EFN_TB_SET_COLUMN
            ):
                _remove(_get_name(t, t.inputs[1]))
                _append(t.inputs[2])

        return list(remove_cols), list(append_cols)

    def dump_serving_pb(
        self, name: str
    ) -> Tuple[compute_trace_pb2.ComputeTrace, pa.Schema, pa.Schema]:
        dag_pb = compute_trace_pb2.ComputeTrace()
        dag_pb.name = name

        dag_tracers = self._flatten_dag()

        dag_input_schema = dag_tracers[0].output_schema
        dag_output_schema = self.output_schema

        trace_output_id_map = {}
        trace_output_id_count = 1  # 0 is dag input
        for t in dag_tracers:
            if t.inputs is None:
                # dag input point
                assert t is dag_tracers[0], "only allow one input in dag"
                trace_output_id_map[t] = 0
            else:
                assert t not in trace_output_id_map
                inputs_pb = []
                assert len(t.inputs) > 0
                for i in t.inputs:
                    input_pb = compute_trace_pb2.FunctionInput()
                    if isinstance(i, _Tracer):
                        assert i in trace_output_id_map
                        input_pb.data_id = trace_output_id_map[i]
                    elif isinstance(i, int) or isinstance(i, np.integer):
                        input_pb.custom_scalar.CopyFrom(
                            compute_trace_pb2.Scalar(i64=int(i))
                        )
                    elif isinstance(i, float) or isinstance(i, np.floating):
                        input_pb.custom_scalar.CopyFrom(
                            compute_trace_pb2.Scalar(d=float(i))
                        )
                    elif isinstance(i, str):
                        input_pb.custom_scalar.CopyFrom(compute_trace_pb2.Scalar(s=i))
                    else:
                        raise AttributeError(f"Unknown type input {i}")

                    inputs_pb.append(input_pb)

                trace_pb = compute_trace_pb2.FunctionTrace(
                    name=t.operate,
                    option_bytes=self.options if self.options is not None else b"",
                    inputs=inputs_pb,
                    output=compute_trace_pb2.FunctionOutput(
                        data_id=trace_output_id_count,
                    ),
                )
                trace_output_id_map[t] = trace_output_id_count
                trace_output_id_count += 1
                dag_pb.func_traces.append(trace_pb)

        return (dag_pb, dag_input_schema, dag_output_schema)


class Array:
    def __init__(self, arrow: pa.ChunkedArray, trace: _Tracer):
        assert isinstance(arrow, pa.ChunkedArray)
        assert isinstance(trace, _Tracer)
        self._arrow = arrow
        self._trace = trace

    @property
    def dtype(self) -> pa.DataType:
        return self._arrow.type


class Table:
    def __init__(self, table: pa.Table, trace: _Tracer):
        assert isinstance(table, pa.Table)
        assert isinstance(trace, _Tracer)
        self._trace = trace
        self._table = table

    @staticmethod
    def schema_check(schema: pa.Schema):
        for dt in schema.types:
            assert (
                pa.types.is_boolean(dt)
                or pa.types.is_floating(dt)
                or pa.types.is_integer(dt)
                or pa.types.is_string(dt)
            ), f"only support bool/float/int/str, got {dt}"
            assert not pa.types.is_float16(dt), "not support float16 for now"

    @staticmethod
    def from_pyarrow(table: pa.Table):
        assert isinstance(table, pa.Table)
        Table.schema_check(table.schema)
        return Table(
            table,
            _Tracer("", output_type=_TracerType.TABLE, output_schema=table.schema),
        )

    @staticmethod
    def from_pandas(pd_: pd.DataFrame):
        assert isinstance(pd_, pd.DataFrame)
        table = pa.Table.from_pandas(pd_)
        Table.schema_check(table.schema)
        return Table(
            table,
            _Tracer("", output_type=_TracerType.TABLE, output_schema=table.schema),
        )

    @staticmethod
    def from_schema(schema: Dict[str, np.dtype]):
        assert isinstance(schema, dict)

        pydist = {}

        for name, dtype in schema.items():
            if dtype == object or dtype == str:
                mock_data = ""
            else:
                mock_data = dtype()
            pydist[name] = mock_data

        table = pa.Table.from_pylist([pydist])
        Table.schema_check(table.schema)
        return Table(
            table,
            _Tracer("", output_type=_TracerType.TABLE, output_schema=table.schema),
        )

    def to_pandas(self) -> pd.DataFrame:
        Table.schema_check(self._table.schema)
        return self._table.to_pandas()

    def dump_serving_pb(
        self, name
    ) -> Tuple[compute_trace_pb2.ComputeTrace, pa.Schema, pa.Schema]:
        Table.schema_check(self._trace.output_schema)
        return self._trace.dump_serving_pb(name)

    def dump_runner(self) -> _TraceRunner:
        Table.schema_check(self._trace.output_schema)
        return self._trace.dump_runner()

    def column_changes(self) -> Tuple[List, List]:
        return self._trace.column_changes()

    # subset of pyarrow.table
    @property
    def column_names(self) -> List[str]:
        return self._table.column_names

    @property
    def shape(self) -> Tuple[int, int]:
        return self._table.shape

    def column(self, i: Union[str, int]) -> Array:
        assert isinstance(i, (str, int))

        if isinstance(i, str):
            assert i in self.column_names
            i = self.column_names.index(i)

        arrow = self._table.column(i)
        tracer = _Tracer(
            compute_trace_pb2.ExtendFunctionName.Name(compute_trace_pb2.EFN_TB_COLUMN),
            output_type=_TracerType.ARROW,
            inputs=[self._trace, i],
        )

        return Array(arrow, tracer)

    def add_column(self, i: int, name: str, array: Array) -> Table:
        assert isinstance(i, int)
        assert isinstance(name, str)
        assert isinstance(array, Array)

        table = self._table.add_column(i, name, array._arrow)
        tracer = _Tracer(
            compute_trace_pb2.ExtendFunctionName.Name(
                compute_trace_pb2.EFN_TB_ADD_COLUMN
            ),
            output_type=_TracerType.TABLE,
            inputs=[self._trace, i, name, array._trace],
            output_schema=table.schema,
        )

        return Table(table, tracer)

    def append_column(self, name: str, array: Array) -> Table:
        return self.add_column(self.shape[1], name, array)

    def remove_column(self, i: Union[int, str]) -> Table:
        assert isinstance(i, (int, str))

        if isinstance(i, str):
            assert i in self.column_names
            i = self.column_names.index(i)

        table = self._table.remove_column(i)
        tracer = _Tracer(
            compute_trace_pb2.ExtendFunctionName.Name(
                compute_trace_pb2.EFN_TB_REMOVE_COLUMN
            ),
            output_type=_TracerType.TABLE,
            inputs=[self._trace, i],
            output_schema=table.schema,
        )

        return Table(table, tracer)

    def set_column(self, i: int, name: str, array: Array) -> Table:
        assert isinstance(i, int)
        assert isinstance(name, str)
        assert isinstance(array, Array)

        table = self._table.set_column(i, name, array._arrow)
        tracer = _Tracer(
            compute_trace_pb2.ExtendFunctionName.Name(
                compute_trace_pb2.EFN_TB_SET_COLUMN
            ),
            output_type=_TracerType.TABLE,
            inputs=[self._trace, i, name, array._trace],
            output_schema=table.schema,
        )

        return Table(table, tracer)
