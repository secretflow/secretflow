from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa

from secretflow.spec.v1 import compute_trace_pb2


class _TracerType(Enum):
    TABLE = 1
    ARROW = 2


class _Tracer:
    def __init__(
        self,
        operate: str,
        output_type: _TracerType,
        inputs: List[Union[_Tracer, int, float, str]] = None,
        options: bytes = None,
        output_schema: pa.Schema = None,
    ):
        self.operate = operate
        self.output_type = output_type
        self.inputs = inputs
        self.options = options
        self.output_schema = output_schema

    def dump(
        self, name: str
    ) -> Tuple[compute_trace_pb2.ComputeTrace, pa.Schema, pa.Schema]:
        dag_pb = compute_trace_pb2.ComputeTrace()
        dag_pb.name = name

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

        dag_tracers.reverse()
        assert dag_tracers[0].output_type is _TracerType.TABLE
        assert dag_tracers[0].inputs is None
        dag_input_schema = dag_tracers[0].output_schema
        dag_output_schema = self.output_schema

        trace_output_id_map = {}
        trace_output_id_count = 1  # 0 is dag input
        for t in dag_tracers:
            if t.inputs is None:
                # dag input point
                assert t is dag_tracers[0], "only allow one input in dag"
                trace_output_id_map[t] = 0
            elif t not in trace_output_id_map:
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


class Table:
    def __init__(self, table: pa.Table, trace: _Tracer):
        assert isinstance(table, pa.Table)
        assert isinstance(trace, _Tracer)
        self._trace = trace
        self._table = table

    @staticmethod
    def from_pyarrow(table: pa.Table):
        assert isinstance(table, pa.Table)
        return Table(
            table,
            _Tracer("", output_type=_TracerType.TABLE, output_schema=table.schema),
        )

    @staticmethod
    def from_pandas(pd_: pd.DataFrame):
        assert isinstance(pd_, pd.DataFrame)
        table = pa.Table.from_pandas(pd_)
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
        return Table(
            table,
            _Tracer("", output_type=_TracerType.TABLE, output_schema=table.schema),
        )

    def to_pandas(self) -> pd.DataFrame:
        return self._table.to_pandas()

    def dump_tracer(
        self, name
    ) -> Tuple[compute_trace_pb2.ComputeTrace, pa.Schema, pa.Schema]:
        return self._trace.dump(name)

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
