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


import pandas as pd
import pyarrow as pa

from secretflow.spec.v1.data_pb2 import DistData, SystemInfo
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab, Table

from ..common.types import Output
from ..common.utils import to_attribute, to_type_str
from .base import DistDataType


class Reporter:
    def __init__(self, name: str = "", desc: str = "") -> None:
        self._name = name
        self._desc = desc
        self._tabs = []

    @staticmethod
    def get_description(attrs: dict) -> str:
        return attrs["description"] if attrs and "description" in attrs else ""

    @staticmethod
    def set_description(f: pa.Field | pd.Series, desc: str):
        if isinstance(f, pa.Field):
            if f.metadata is None:
                f.metadata = {b"description": desc}
            else:
                f.metadata[b"description"] = desc
        elif isinstance(f, pd.Series):
            if f.attrs is None:
                f.attrs = {b"description": desc}
            else:
                f.attrs[b"description"] = desc
        else:
            raise ValueError(f"unsupport type, {type(f)}, {desc}")

    @staticmethod
    def to_table(df: pd.DataFrame | pa.Table, name: str = "", desc: str = "", prefix: str = "") -> Table:  # type: ignore
        headers, rows = [], []
        if isinstance(df, pd.DataFrame):
            for k in df.columns:
                dtype = to_type_str(df[k].dtype)
                hdesc = Reporter.get_description(df[k].attrs)
                headers.append(Table.HeaderItem(name=k, desc=hdesc, type=dtype))

            for index in df.index:
                items = []
                for k in df.columns:
                    value = df.at[index, k]
                    items.append(to_attribute(value))
                rows.append(Table.Row(name=f'{prefix}{index}', items=items))
        elif isinstance(df, pa.Table):
            for f in df.schema:
                dtype = to_type_str(f.type)
                hdesc = Reporter.get_description(f.metadata)
                headers.append(Table.HeaderItem(name=k, desc=hdesc, type=dtype))

            for i in range(df.num_rows):
                pa_row = df.slice(i, 1)
                items = []
                for column in df.column_names:
                    value = pa_row[column][0].as_py()
                    items.append(to_attribute(value))
                rows.append(Table.Row(name=f'{prefix}{i}', items=items))

        return Table(name=name, desc=desc, headers=headers, rows=rows)

    @staticmethod
    def to_descriptions(values: dict[str, int | float | bool | str], name: str = "", desc: str = "") -> Descriptions:  # type: ignore
        items = [
            Descriptions.Item(name=k, type=to_type_str(v), value=to_attribute(v))
            for k, v in values.items()
        ]
        return Descriptions(name=name, desc=desc, items=items)

    @staticmethod
    def to_div(content: Table | Descriptions | Div, name: str = None, desc: str = None) -> Div:  # type: ignore
        if isinstance(content, Table):
            child = Div.Child(type='table', table=content)
        elif isinstance(content, Descriptions):
            child = Div.Child(type='descriptions', descriptions=content)
        elif isinstance(content, Div):
            child = Div.Child(type='div', div=content)
        else:
            raise ValueError(f"invalid div child type, {type(content)}")
        return Div(name=name, desc=desc, children=[child])

    @staticmethod
    def to_div_child(content: Table | Descriptions | Div) -> Div.Child:  # type: ignore
        if isinstance(content, Table):
            return Div.Child(type='table', table=content)
        elif isinstance(content, Descriptions):
            return Div.Child(type='descriptions', descriptions=content)
        elif isinstance(content, Div):
            return Div.Child(type='div', div=content)
        else:
            raise ValueError(f"invalid div child type, {type(content)}")

    def add_tab(self, content: Div | Table | pd.DataFrame | pa.Table | Descriptions | dict[str, object] | list[Div], name: str = None, desc: str = None):  # type: ignore
        divs: list[Div] = []  # type: ignore
        if isinstance(content, Div):
            divs.append(content)
        elif isinstance(content, (pd.DataFrame, pa.Table)):
            table = self.to_table(content)
            divs.append(Div(children=[Div.Child(type="table", table=table)]))
        elif isinstance(content, Table):
            divs.append(Div(children=[Div.Child(type="table", table=content)]))
        elif isinstance(content, dict):
            descriptions = self.to_descriptions(content)
            divs.append(
                Div(
                    children=[Div.Child(type="descriptions", descriptions=descriptions)]
                )
            )
        elif isinstance(content, Descriptions):
            divs.append(
                Div(children=[Div.Child(type="descriptions", descriptions=content)])
            )
        elif isinstance(content, list) and isinstance(content[0], Div):
            divs = content
        else:
            raise ValueError(f"unsupport type {type(content)}")
        self._tabs.append(Tab(name=name, desc=desc, divs=divs))

    def report(self) -> Report:  # type: ignore
        return Report(name=self._name, desc=self._desc, tabs=self._tabs)

    def dump(self, uri: str, system_info: SystemInfo) -> DistData:
        dd = DistData(
            name=uri,
            type=str(DistDataType.REPORT),
            system_info=system_info,
        )
        if self._tabs:
            dd.meta.Pack(self.report())
        return dd

    def dump_to(self, out: Output, system_info: SystemInfo):
        out.data = self.dump(out.uri, system_info)
