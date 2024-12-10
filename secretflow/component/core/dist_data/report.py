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


from secretflow.spec.v1.data_pb2 import DistData, SystemInfo
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab, Table

from ..utils import to_attribute, to_type
from .base import DistDataType


class Reporter:
    def __init__(
        self,
        name: str = "",
        desc: str = "",
        tabs: list[Tab] = None,
        system_info: SystemInfo = None,
        type: str = DistDataType.REPORT,
    ) -> None:
        self._name = name
        self._desc = desc
        self._tabs = tabs if tabs else []
        self._system_info = system_info
        self._type = str(type)

    @staticmethod
    def from_distdata(dd: DistData) -> "Reporter":
        if dd.meta:
            report = Report()
            dd.meta.Unpack(report)
            return Reporter(
                report.name, report.desc, report.tabs, dd.system_info, type=dd.type
            )

        return Report(dd.name)

    def to_distdata(self) -> DistData:
        dd = DistData(name=self._name, type=self._type, system_info=self._system_info)
        meta = self.report()
        if meta:
            dd.meta.Pack(meta)

        return dd

    def report(self) -> Report:
        if self._tabs:
            return Report(name=self._name, desc=self._desc, tabs=self._tabs)
        return None

    def add_tab(
        self,
        obj: list[Div] | Div | Table | Descriptions | dict,
        name: str = None,
        desc: str = None,
    ):
        divs: list[Div] = []
        if isinstance(obj, list):
            assert all(
                isinstance(item, Div) for item in obj
            ), f"all item should be instance of Div, {obj}"
            divs = obj
        elif isinstance(obj, Div):
            divs.append(obj)
        else:
            child = self.build_div_child(obj)
            divs.append(Div(children=[child]))

        self._tabs.append(Tab(name=name, desc=desc, divs=divs))

    @staticmethod
    def build_table(
        obj: dict,
        name: str = None,
        desc: str = None,
        columns: dict[str, Table.HeaderItem | str] = None,
        index: list[str] = None,
        prefix: str = "",
    ) -> Table:
        '''
        name: table name
        desc: table description
        columns: columns header info, if type of dict value is str, it represents column description
        index: row index
        prefix: row index name prefix
        '''
        pb_headers, pb_rows = [], []
        df = _to_dict(obj)
        for col_name in df.keys():
            dtype = _to_type_str(df[col_name][0])
            if columns and col_name in columns:
                v = columns[col_name]
                header = (
                    v
                    if isinstance(v, Table.HeaderItem)
                    else Table.HeaderItem(name=col_name, desc=v, type=dtype)
                )
            else:
                header = Table.HeaderItem(name=col_name, desc="", type=dtype)
            pb_headers.append(header)

        row_size = len(next(iter(df.values())))
        for idx in range(row_size):
            items = []
            for k in df.keys():
                value = df[k][idx]
                items.append(to_attribute(value))
            idx_name = index[idx] if index and idx < len(index) else str(idx)
            row_name = f"{prefix}{idx_name}"
            pb_rows.append(Table.Row(name=row_name, items=items))
        return Table(name=name, desc=desc, headers=pb_headers, rows=pb_rows)

    @staticmethod
    def build_descriptions(
        values: dict[str, int | float | bool | str], name: str = None, desc: str = None
    ) -> Descriptions:
        items = [
            Descriptions.Item(name=k, type=_to_type_str(v), value=to_attribute(v))
            for k, v in values.items()
        ]
        return Descriptions(name=name, desc=desc, items=items)

    @staticmethod
    def build_div_child(obj: Table | Descriptions | Div | dict) -> Div.Child:
        if isinstance(obj, Table):
            return Div.Child(type="table", table=obj)
        elif isinstance(obj, Descriptions):
            return Div.Child(type="descriptions", descriptions=obj)
        elif isinstance(obj, Div):
            return Div.Child(type="div", div=obj)
        else:
            obj = _to_dict(obj)
            if _is_table_dict(obj):
                table = Reporter.build_table(obj)
                return Div.Child(type="table", table=table)
            else:
                descriptions = Reporter.build_descriptions(obj)
                return Div.Child(type="descriptions", descriptions=descriptions)

    @staticmethod
    def build_div(
        obj: Table | Descriptions | Div | dict, name: str = None, desc: str = None
    ) -> Div:
        child = Reporter.build_div_child(obj)
        return Div(name=name, desc=desc, children=[child])


def _to_type_str(dt) -> str:
    dt = to_type(dt)
    return dt.__name__


def _is_table_dict(value: dict) -> bool:
    if not value:
        return False

    # {"A": [1,2,3], "B": [0.1,0.2,0.3]}
    sizes = set()
    for x in value.values():
        if not isinstance(x, list):
            return False
        sizes.add(len(x))

    if len(sizes) != 1 or list(sizes)[0] < 1:
        return False

    return True


def _to_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj

    # only support pd.DataFrame
    if hasattr(obj, "to_dict"):
        method = getattr(obj, "to_dict")
        if callable(method):
            return method(orient="list")

    raise ValueError(f"unsupport type, {type(obj)}")
