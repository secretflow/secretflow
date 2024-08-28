# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from dataclasses import dataclass

import pytest

from secretflow.component.core import (
    Component,
    Context,
    Definition,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    UnionGroup,
)


@dataclass
class UnionOption1:
    state: int = Field.attr(desc="state", default=1)


@dataclass
class UnionOption2:
    field1: int = Field.attr(desc="field1")


@dataclass
class UnionField(UnionGroup):
    option1: UnionOption1 = Field.struct_attr(desc="option1")
    option2: UnionOption2 = Field.struct_attr(desc="option2")
    option3: str = Field.selection_attr(desc="option3, selection attr")
    option4: int = Field.attr(desc="option4, int attr")


@dataclass
class StructField:
    field1: int = Field.attr(default=1)
    field2: float = Field.attr(default=1.0, bound_limit=Interval.closed(0, 100))


class StructNoDataclass:
    field1: int = Field.attr(default=1)


class DemoComponent(Component):
    """
    demo component
    """

    class StructInner:
        field1: int = Field.attr(default=1)

    int_fd: int = Field.attr(default=1, choices=[0, 1])
    float_fd: float = Field.attr(default=1.0, bound_limit=Interval.closed(0, 2))
    bool_fd: bool = Field.attr(default=True)
    ints_fd: list[int] = Field.attr(default=[1, 2, 3], list_limit=Interval.closed(1, 3))
    floats_fd: list[float] = Field.attr(
        default=[1.0, 2.0, 3.0], list_limit=Interval.closed(1, 3)
    )
    bools_fd: list[bool] = Field.attr(default=[True, False])
    strs_fd: list[str] = Field.attr(default=["a", "b", "c"])
    struct_fd: StructField = Field.struct_attr(desc="struct field")
    struct_no_dc: StructNoDataclass = Field.struct_attr()
    struct_inner: StructInner = Field.struct_attr()
    union_fd: UnionField = Field.union_attr(desc="union field")
    column: str = Field.table_column_attr("input_fd", desc="input table column")
    drop_first: bool = Field.attr(desc="drop_first", minor_max=0)
    drop: str = Field.attr(choices=["first", "mod", "no_drop"], minor_min=1)
    input_fd: Input = Field.input(desc="input", types=[DistDataType.INDIVIDUAL_TABLE])
    output_fd: Output = Field.output(
        desc="output", types=[DistDataType.INDIVIDUAL_TABLE]
    )

    def __post__init__(self) -> None:
        # upgrade param and verify
        if self.is_supported(0):
            self.drop = "first" if self.drop_first else "no_drop"

    def evaluate(self, ctx: Context):
        with ctx.trace_running():
            print(self.int_fd, self.float_fd)


def test_definition():
    comp1 = DemoComponent(
        int_fd=1,
        struct_fd=StructField(field1=2),
        struct_no_dc=StructNoDataclass(field1=3),
        struct_inner=DemoComponent.StructInner(field1=4),
    )
    assert (
        comp1.int_fd == 1
        and comp1.struct_fd.field1 == 2
        and comp1.struct_no_dc.field1 == 3
    )

    args = {
        "int_fd": 1,
        "float_fd": 2.0,
        "bool_fd": True,
        "struct_fd/field1": 11,
        "struct_fd/field2": 12,
        "struct_no_dc/field1": 13,
        "struct_inner/field1": 14,
        "union_fd": "option4",
        "union_fd/option4": 21,
        "drop_first": True,
        "input/input_fd/column": "col1",
        "input_fd": Input(type=str(DistDataType.INDIVIDUAL_TABLE)),
        "output_fd": "test uri",
    }
    args["_minor"] = 0
    comp_def = Definition(DemoComponent, "test", "0.1.0")
    comp: DemoComponent = comp_def.make_component(args)
    assert (
        comp._minor == 0
        and comp.int_fd == 1
        and comp.float_fd == 2.0
        and comp.bool_fd == True
        and comp.struct_fd.field1 == 11
        and comp.struct_fd.field2 == 12
        and comp.struct_no_dc.field1 == 13
        and comp.struct_inner.field1 == 14
        and comp.union_fd.is_selected("option4")
        and comp.union_fd.option4 == 21
        and comp.column == "col1"
    )


def test_union():
    class UnionField(UnionGroup):
        f1: int = Field.attr()
        f2: int = Field.attr()

    class UnionComponent(Component):
        uf: UnionField = Field.union_attr()

        def evaluate(self, ctx: Context): ...

    args = {"uf": "f1", "uf/f1": 1}
    args["_minor"] = 0
    comp_def = Definition(UnionComponent, "test", "0.1.0")
    comp: UnionComponent = comp_def.make_component(args)
    assert comp.uf.is_selected("f1") and comp.uf.f1 == 1

    # uf default use f1
    args = {"uf/f1": 1}
    args["_minor"] = 0
    comp_def.make_component(args)

    # no uf/f1
    args = {"uf": "f1"}
    args["_minor"] = 0
    with pytest.raises(Exception) as exc_info:
        comp_def.make_component(args)
    logging.warning(f"Caught expected Exception: {exc_info}")

    # unused uf/f2
    args = {"uf": "f1", "uf/f1": 1, "uf/f2": 2}
    args["_minor"] = 0
    with pytest.raises(Exception) as exc_info:
        comp_def.make_component(args)
    logging.warning(f"Caught expected Exception: {exc_info}")


def test_version():
    class VersionComponent(Component):
        v0: int = Field.attr()
        v1: int = Field.attr(minor_min=1, minor_max=1)
        v2: int = Field.attr(minor_min=2)

        input1: Input = Field.input(minor_min=0, minor_max=0)
        input2: Input = Field.input(minor_min=1)

        output1: Output = Field.output(minor_min=0, minor_max=0)
        output2: Output = Field.output(minor_min=1)

        def evaluate(self, ctx: Context) -> None: ...

    definition = Definition(VersionComponent, "test", "0.2.0")
    comp_def = definition.component_def
    assert len(comp_def.attrs) == 2 and comp_def.attrs[1].name == "v2"
    assert len(comp_def.inputs) == 1 and len(comp_def.outputs) == 1

    args = {"v0": 0, "input1": Input(), "output1": Output(uri="o1")}
    args["_minor"] = 0
    comp: VersionComponent = definition.make_component(args)
    assert comp.v0 == 0 and comp.v1 is None and comp.v2 is None

    args = {"v0": 0, "v1": 1, "input2": Input(), "output2": Output(uri="o1")}
    args["_minor"] = 1
    comp: VersionComponent = definition.make_component(args)
    assert comp.v0 == 0 and comp.v1 == 1 and comp.v2 is None

    args = {"v0": 0, "v2": 2, "input2": Input(), "output2": Output(uri="o2")}
    args["_minor"] = 2
    comp: VersionComponent = definition.make_component(args)
    assert comp.v0 == 0 and comp.v2 == 2 and comp.output2.uri == "o2"

    bad_cases = [
        {"_minor": 0, "input1": Input(), "output1": Output(uri="o1")},  # no v0
        {
            "_minor": 2,
            "v1": 1,
            "v0": 0,
            "v2": 2,
            "input2": Input(),
            "output2": Output(uri="o2"),
        },  # no need v1
    ]

    for case in bad_cases:
        with pytest.raises(Exception):
            definition.make_component(case)


def test_inherit():
    class Parent(Component):
        f1: int = Field.attr()

    class Child(Parent):
        f2: int = Field.attr()

        def evaluate(self, ctx: Context) -> None:
            with ctx.trace_running():
                print(self.f1, self.f2)

    child = Child(f1=1, f2=2)
    assert child.f1 == 1 and child.f2 == 2

    comp_def = Definition(Child, "test", "0.0.1")
    args = {"_minor": 0, "f1": 1, "f2": 2}
    comp: Child = comp_def.make_component(args)
    assert comp.f1 == 1 and comp.f2 == 2


def test_default():
    class TestDefaultComponent(Component):
        f1: int = Field.attr(default=1)
        f2: int = Field.attr()

        def evaluate(self, ctx: Context) -> None: ...

    dc = TestDefaultComponent(f2=2)
    assert dc.f1 == 1 and dc.f2 == 2

    comp_def = Definition(TestDefaultComponent, "test", "0.0.1")
    args = {"_minor": 0, "f2": 2}
    comp: TestDefaultComponent = comp_def.make_component(args)
    assert comp.f1 == 1 and comp.f2 == 2
