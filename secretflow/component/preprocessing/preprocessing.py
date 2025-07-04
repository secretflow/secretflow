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

import abc
import inspect
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import pyarrow as pa
from google.protobuf import json_format
from secretflow_serving_lib import compute_trace_pb2
from secretflow_spec.v1.data_pb2 import DistData, SystemInfo

import secretflow.compute as sc
from secretflow.component.core import (
    BINNING_RULE_MAX,
    PREPROCESSING_RULE_MAX,
    CompVDataFrame,
    CompVDataFrameReader,
    CompVDataFrameWriter,
    Context,
    Definition,
    DistDataType,
    Input,
    Model,
    Output,
    Registry,
    ServingBuilder,
    ServingNode,
    ServingOp,
    ServingPhase,
    Version,
    VTable,
    VTableFieldKind,
    VTableUtils,
    uuid4,
)
from secretflow.device import PYU, PYUObject, reveal

RULE_TYPE_NAME = "rule_type"
RULE_HASH_NAME = "model_hash"


class IRunner(abc.ABC):
    """
    IRunner is used to unify the preprocessing model replay and model export functions, with an interface similar to sc.TraceRunner.
    The ArrowRunner class is implemented by default, which is used for saving in the sc.TraceRunner format.
    For specific requirements, you can inherit from IRunner to create a custom implementation.
    For example, if binning requires modifications to the model, you can implement a custom BinningRunner.
    """

    @abc.abstractmethod
    def get_input_features(self) -> list[str]:
        pass

    @abc.abstractmethod
    def run(self, pt: pa.Table) -> pa.Table:
        """
        run is used to replay model
        """
        pass

    @abc.abstractmethod
    def column_changes(self, input_schema: pa.Schema) -> tuple[list, list, list]:
        """
        Only for compatibility with model_export
        """
        pass

    @abc.abstractmethod
    def dump_serving_pb(
        self, name: str, input_schema: pa.Schema
    ) -> tuple[compute_trace_pb2.ComputeTrace, pa.Schema, pa.Schema]:
        """
        dump_serving_pb is used to export the model for serving
        """
        pass


def build_schema(
    new_tbl: pa.Table, old_tbl: pa.Table, add_kinds: dict[str, VTableFieldKind]
) -> pa.Table:
    old_column_names = set(old_tbl.column_names)
    fields = []
    for f in new_tbl.schema:
        if f.name in old_column_names:
            old = old_tbl.field(f.name)
            f = VTableUtils.pa_field_from(f.name, f.type, old)
            fields.append(f)
        else:
            kind = VTableFieldKind.FEATURE
            if f.name in add_kinds:
                kind = add_kinds[f.name]
            f = VTableUtils.pa_field(f.name, f.type, kind)
            fields.append(f)
    return pa.table(new_tbl.columns, schema=pa.schema(fields))


@dataclass
class ArrowRunner(IRunner):
    trace: sc.TraceRunner
    kinds: dict[str, VTableFieldKind]

    @staticmethod
    def from_table(tbl: sc.Table, input_columns: list[str]) -> "ArrowRunner":
        trace_runner = tbl.dump_runner()
        out_schema = VTableUtils.from_arrow_schema(tbl.to_table().schema)
        add_kinds = {
            n: k for n, k in out_schema.kinds.items() if n not in set(input_columns)
        }

        return ArrowRunner(trace_runner, add_kinds)

    def column_changes(self, input_schema: pa.Schema) -> tuple[list, list, list]:
        return self.trace.column_changes()

    def get_input_features(self) -> list[str]:
        return self.trace.get_input_features()

    def run(self, df: pa.Table) -> pa.Table:
        new_df = self.trace.run(df)
        return build_schema(new_df, df, self.kinds)

    def dump_serving_pb(
        self, name: str, input_schema: pa.Schema
    ) -> tuple[compute_trace_pb2.ComputeTrace, pa.Schema, pa.Schema]:
        return self.trace.dump_serving_pb(name)


def _merge(remain_data: pa.Table, out_data: pa.Table, order: list[str]) -> pa.Table:
    schema = pa.schema(list(remain_data.schema) + list(out_data.schema))
    new_data = pa.table(remain_data.columns + out_data.columns, schema=schema)

    # sort by original order
    new_columns_map = {name: idx for idx, name in enumerate(new_data.schema.names)}
    columns = []
    for name in order:
        if name in new_columns_map:
            columns.append(name)
            del new_columns_map[name]
    columns.extend(list(new_columns_map.keys()))
    new_data = new_data.select(columns)
    return new_data


class PreprocessingMixin:
    @staticmethod
    def build_model(
        component_id: str,
        model_type: DistDataType,
        model_version: Version,
        objs: list[PYUObject],
        public_info: Any | None = None,
        system_info: SystemInfo = None,
    ) -> Model:
        if not public_info:
            public_info = {}

        uuid = uuid4(objs[0].device)
        return Model(
            name=component_id,
            type=model_type,
            version=model_version,
            objs=objs,
            public_info=public_info,
            attributes={RULE_HASH_NAME: uuid},
            system_info=system_info,
        )

    @staticmethod
    def get_version_max(model_type: DistDataType | str) -> Version:
        model_type = DistDataType(model_type)
        if model_type == DistDataType.BINNING_RULE:
            return BINNING_RULE_MAX
        elif model_type == DistDataType.PREPROCESSING_RULE:
            return PREPROCESSING_RULE_MAX
        else:
            raise ValueError(f"invalid model_type<{model_type}>")

    def model_info(self) -> tuple[DistDataType, Version]:
        return DistDataType.PREPROCESSING_RULE, PREPROCESSING_RULE_MAX

    def fit(
        self,
        ctx: Context,
        out_rule: Output,
        input: Input | VTable,
        fn: (
            Callable[[sc.Table], sc.Table | IRunner]
            | Callable[[sc.Table, object], sc.Table | IRunner]
        ),
        extras: dict[str, PYUObject | Any] = None,
    ) -> Model:
        """
        build a rule by a funtion and input schema

        NOTE: the function SHOULD not rely on the input data,table statistical info can be passed through 'extras'.
        """
        signature = inspect.signature(fn)
        is_one_param = len(signature.parameters) == 1

        def _fit(trans_data: sc.Table, extra: object) -> IRunner:
            if is_one_param:
                out_data = fn(trans_data)
            else:
                out_data = fn(trans_data, extra)

            assert isinstance(out_data, (sc.Table, IRunner))

            if isinstance(out_data, sc.Table):
                runner = ArrowRunner.from_table(out_data, in_tbl.column_names)
            else:
                runner = out_data

            return runner

        tbl = input if isinstance(input, VTable) else VTable.from_distdata(input)
        runner_objs = []
        for p in tbl.parties.values():
            pa_schema = VTableUtils.to_arrow_schema(p.schema)
            in_tbl = sc.Table.from_schema(pa_schema)
            extra = extras.get(p.party) if extras else None
            out_runner = PYU(p.party)(_fit)(in_tbl, extra)
            runner_objs.append(out_runner)

        defi = Registry.get_definition_by_class(self)
        model_type, model_version = self.model_info()
        rule = self.build_model(
            defi.component_id,
            model_type,
            model_version,
            runner_objs,
            None,
            tbl.system_info,
        )
        ctx.dump_to(rule, out_rule)
        return rule

    def transform(
        self,
        ctx: Context,
        output: Output,
        input: Input | VTable | CompVDataFrame,
        rule: Model,
        streaming: bool = True,
    ):
        def _transform(data: pa.Table, runner: IRunner) -> pa.Table:
            trans_columns = list(runner.get_input_features())
            if len(trans_columns) == 0:
                return data

            assert set(trans_columns).issubset(
                set(data.column_names)
            ), f"can not find rule keys {trans_columns} in dataset columns {data.column_names}"

            trans_data = data.select(trans_columns)
            remain_data = data.drop(trans_columns)
            out_data = runner.run(trans_data)
            new_data = _merge(remain_data, out_data, data.column_names)
            return new_data

        rule_objs = {}
        for obj in rule.objs:
            assert isinstance(obj, PYUObject)
            rule_objs[obj.device.party] = obj

        def apply(df: CompVDataFrame) -> CompVDataFrame:
            out_df = CompVDataFrame({}, input.system_info)
            for pyu, p in df.partitions.items():
                if pyu.party in rule_objs:
                    out_data = pyu(_transform)(p.data, rule_objs[pyu.party])
                else:
                    out_data = p.obj
                out_df.set_data(out_data)
            return out_df

        if streaming:
            assert not isinstance(input, CompVDataFrame)
            reader = CompVDataFrameReader(ctx.storage, ctx.tracer, input)
            writer = CompVDataFrameWriter(ctx.storage, ctx.tracer, output.uri)
            for df in reader:
                with ctx.trace_running():
                    out_df = apply(df)
                writer.write(out_df)
            reader.close()
            writer.close()
            writer.dump_to(output)
        else:
            df = input if isinstance(input, CompVDataFrame) else ctx.load_table(input)
            out_df = apply(df)
            ctx.dump_to(out_df, output)

    def fit_transform(
        self,
        ctx: Context,
        out_rule: Output,
        out_ds: Output,
        in_tbl: VTable,
        trans_tbl: VTable,
        fn: Callable[[sc.Table], sc.Table],
    ) -> None:
        """
        Only for compatibility with legacy code, the new implementation recommends using the fit+transform combination.
        """

        def _fit_transform(
            data: pa.Table, columns: list[str]
        ) -> Tuple[pa.Table, IRunner, Exception]:
            trans_data = data.select(columns)
            remain_data = data.drop(columns)

            try:
                out_data = fn(sc.Table.from_pyarrow(trans_data))
            except Exception as e:
                traceback.print_exc()
                return None, None, e

            assert isinstance(out_data, sc.Table)

            runner = ArrowRunner.from_table(out_data, trans_data.column_names)

            new_data = _merge(remain_data, out_data.to_table(), data.column_names)

            return new_data, runner, None

        out_df = CompVDataFrame({}, in_tbl.system_info)
        out_runners = []
        df = ctx.load_table(in_tbl)
        for pyu, p in df.partitions.items():
            if pyu.party not in trans_tbl.parties:
                out_df.partitions[pyu] = p
                continue
            columns = trans_tbl.parties[pyu.party].schema.names
            data, runner, err = pyu(_fit_transform)(p.data, columns)
            err = reveal(err)
            if err is not None:
                raise err

            out_runners.append(runner)
            out_df.set_data(data)

        ctx.dump_to(out_df, out_ds)

        defi = Registry.get_definition_by_class(self)
        model_type, model_version = self.model_info()
        model = self.build_model(
            defi.component_id,
            model_type,
            model_version,
            out_runners,
            None,
            in_tbl.system_info,
        )
        ctx.dump_to(model, out_rule)

    def do_export(
        self,
        ctx: Context,
        builder: ServingBuilder,
        input_dd: DistData,
        rules: DistData,
        is_substitution: bool = False,
    ):
        input_vtbl = VTable.from_distdata(input_dd)
        pyus = {p.party: p for p in builder.pyus}
        version_max = self.get_version_max(rules.type)
        model = ctx.load_model(rules, version=version_max, pyus=pyus)
        runner_objs = {r.device: r for r in model.objs}

        try:
            _, comp_name, _ = Definition.parse_id(model.name)
            if is_substitution:
                node_name = f"{comp_name}_substitution_{builder.max_id()}"
            else:
                node_name = f"{comp_name}_{builder.max_id()}"
        except:
            defi = Registry.get_definition_by_class(self)
            node_name = f"{defi.name}_{builder.max_id()}"
        node = ServingNode(
            node_name,
            op=ServingOp.ARROW_PROCESSING,
            phase=ServingPhase.PREPROCESSING,
        )

        def _dump_runner(
            input_schema: pa.Schema, runner: IRunner, node_name: str
        ) -> Tuple[bytes, pa.Schema, pa.Schema, bytes, bytes]:
            features = runner.get_input_features()
            input_fields = [input_schema.field_by_name(c) for c in features]
            dag_pb, dag_input_schema, dag_output_schema = runner.dump_serving_pb(
                node_name, pa.schema(input_fields)
            )

            dag_json = json_format.MessageToJson(dag_pb, indent=0).encode("utf-8")
            dag_in_schema_bytes = dag_input_schema.serialize().to_pybytes()
            dag_out_schema_bytes = dag_output_schema.serialize().to_pybytes()

            return (
                dag_json,
                dag_input_schema,
                dag_output_schema,
                dag_in_schema_bytes,
                dag_out_schema_bytes,
            )

        for pyu in builder.pyus:
            if pyu not in runner_objs:
                continue
            schema = VTableUtils.to_arrow_schema(input_vtbl.parties[pyu.party].schema)
            runner = runner_objs[pyu]
            dag, in_schema, out_schema, in_schema_bytes, out_schema_bytes = pyu(
                _dump_runner
            )(schema, runner, node_name)
            in_schema = reveal(in_schema)
            out_schema = reveal(out_schema)
            kwargs = ServingNode.build_arrow_processing_kwargs(
                in_schema_bytes, out_schema_bytes, dag
            )
            node.add(pyu, in_schema, out_schema, kwargs)
        builder.add_node(node)
