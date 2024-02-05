import pytest

from secretflow.component.component import Component
from secretflow.component.eval_param_reader import (
    EvalParamError,
    EvalParamReader,
    check_allowed_values,
    check_lower_bound,
    check_table_attr_col_cnt,
    check_upper_bound,
)
from secretflow.spec.v1.component_pb2 import (
    Attribute,
    AttributeDef,
    AttrType,
    ComponentDef,
    IoDef,
)
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def test_check_allowed_values():
    d1 = AttributeDef(
        type=AttrType.AT_FLOAT,
        atomic=AttributeDef.AtomicAttrDesc(
            allowed_values=Attribute(fs=[1.0, 2.0, 3.0])
        ),
    )
    v1 = Attribute(f=1.0)
    assert check_allowed_values(v1, d1)

    v2 = Attribute(f=4.0)
    assert not check_allowed_values(v2, d1)

    d2 = AttributeDef(
        type=AttrType.AT_INT,
        atomic=AttributeDef.AtomicAttrDesc(allowed_values=Attribute(i64s=[1, 2, 3])),
    )

    v3 = Attribute(i64=1)
    assert check_allowed_values(v3, d2)

    d3 = AttributeDef(
        type=AttrType.AT_STRING,
        atomic=AttributeDef.AtomicAttrDesc(
            allowed_values=Attribute(ss=["1", "2", "3"])
        ),
    )

    v4 = Attribute(s="1")
    assert check_allowed_values(v4, d3)


def test_check_lower_bound():
    d1 = AttributeDef(
        type=AttrType.AT_FLOAT,
        atomic=AttributeDef.AtomicAttrDesc(
            lower_bound_enabled=True, lower_bound=Attribute(f=1.0)
        ),
    )

    v1 = Attribute(f=1.0)
    assert not check_lower_bound(v1, d1)

    v2 = Attribute(f=0.5)
    assert not check_lower_bound(v2, d1)

    v3 = Attribute(f=1.5)
    assert check_lower_bound(v3, d1)

    d2 = AttributeDef(
        type=AttrType.AT_FLOAT,
        atomic=AttributeDef.AtomicAttrDesc(
            lower_bound_enabled=True,
            lower_bound_inclusive=True,
            lower_bound=Attribute(f=1.0),
        ),
    )

    assert check_lower_bound(v1, d2)
    assert not check_lower_bound(v2, d2)
    assert check_lower_bound(v3, d2)

    d3 = AttributeDef(
        type=AttrType.AT_INT,
        atomic=AttributeDef.AtomicAttrDesc(
            lower_bound_enabled=True, lower_bound=Attribute(i64=1)
        ),
    )

    v4 = Attribute(i64=1)
    assert not check_lower_bound(v4, d3)

    v5 = Attribute(i64=-1)
    assert not check_lower_bound(v5, d3)

    v6 = Attribute(i64=2)
    assert check_lower_bound(v6, d3)

    d4 = AttributeDef(
        type=AttrType.AT_INT,
        atomic=AttributeDef.AtomicAttrDesc(
            lower_bound_enabled=True,
            lower_bound_inclusive=True,
            lower_bound=Attribute(i64=1),
        ),
    )

    assert check_lower_bound(v4, d4)
    assert not check_lower_bound(v5, d4)
    assert check_lower_bound(v6, d4)


def test_check_upper_bound():
    d1 = AttributeDef(
        type=AttrType.AT_FLOAT,
        atomic=AttributeDef.AtomicAttrDesc(
            upper_bound_enabled=True, upper_bound=Attribute(f=1.0)
        ),
    )

    v1 = Attribute(f=1.0)
    assert not check_upper_bound(v1, d1)

    v2 = Attribute(f=0.5)
    assert check_upper_bound(v2, d1)

    v3 = Attribute(f=1.5)
    assert not check_upper_bound(v3, d1)

    d2 = AttributeDef(
        type=AttrType.AT_FLOAT,
        atomic=AttributeDef.AtomicAttrDesc(
            upper_bound_enabled=True,
            upper_bound_inclusive=True,
            upper_bound=Attribute(f=1.0),
        ),
    )

    assert check_upper_bound(v1, d2)
    assert check_upper_bound(v2, d2)
    assert not check_upper_bound(v3, d2)

    d3 = AttributeDef(
        type=AttrType.AT_INT,
        atomic=AttributeDef.AtomicAttrDesc(
            upper_bound_enabled=True, upper_bound=Attribute(i64=1)
        ),
    )

    v4 = Attribute(i64=1)
    assert not check_upper_bound(v4, d3)

    v5 = Attribute(i64=-1)
    assert check_upper_bound(v5, d3)

    v6 = Attribute(i64=2)
    assert not check_upper_bound(v6, d3)

    d4 = AttributeDef(
        type=AttrType.AT_INT,
        atomic=AttributeDef.AtomicAttrDesc(
            upper_bound_enabled=True,
            upper_bound_inclusive=True,
            upper_bound=Attribute(i64=1),
        ),
    )

    assert check_upper_bound(v4, d4)
    assert check_upper_bound(v5, d4)
    assert not check_upper_bound(v6, d4)


def test_check_table_attr_col_cnt():
    a = Attribute(ss=["a", "b", "c"])

    d1 = IoDef.TableAttrDef(col_min_cnt_inclusive=4)

    assert not check_table_attr_col_cnt(a, d1)

    d2 = IoDef.TableAttrDef(col_max_cnt_inclusive=2)

    assert not check_table_attr_col_cnt(a, d2)

    d3 = IoDef.TableAttrDef(col_min_cnt_inclusive=3, col_max_cnt_inclusive=3)

    assert check_table_attr_col_cnt(a, d3)


def test_node_reader_not_match():
    instance1 = NodeEvalParam(domain="domain_a")
    definition1 = ComponentDef(domain="domain_b")
    with pytest.raises(EvalParamError, match="domain .* does not match."):
        EvalParamReader(instance1, definition1)

    instance2 = NodeEvalParam(domain="domain_a", name="x")
    definition2 = ComponentDef(domain="domain_a", name="y")
    with pytest.raises(EvalParamError, match="name .* does not match."):
        EvalParamReader(instance2, definition2)

    instance3 = NodeEvalParam(domain="domain_a", name="x", version="v1")
    definition3 = ComponentDef(domain="domain_a", name="x", version="v2")
    with pytest.raises(EvalParamError, match="version .* does not match."):
        EvalParamReader(instance3, definition3)


def test_node_reader_duplicate_param_in_instance():
    instance = NodeEvalParam(
        domain="domain_a",
        name="x",
        version="v1",
        attr_paths=["a", "a"],
        attrs=[Attribute(), Attribute()],
    )
    definition = ComponentDef(domain="domain_a", name="x", version="v1")
    with pytest.raises(EvalParamError, match="attr .*is duplicate in node def."):
        EvalParamReader(instance, definition)


def test_node_reader_union_group_selection():
    instance_error = NodeEvalParam(
        domain="domain_a",
        name="x",
        version="v1",
        attr_paths=["a/b/c", "a/b/d", "a/e"],
        attrs=[Attribute(), Attribute(), Attribute()],
    )
    instance_ok = NodeEvalParam(
        domain="domain_a",
        name="x",
        version="v1",
        attr_paths=["a/b/c", "a/b/d"],
        attrs=[Attribute(s="c"), Attribute(s="d")],
    )
    comp = Component(name="x", domain="domain_a", version="v1")
    comp.union_attr_group(
        name="a",
        desc="",
        group=[
            comp.struct_attr_group(
                name="b",
                desc="",
                group=[
                    comp.str_attr(
                        name="c",
                        desc="",
                        is_list=False,
                        is_optional=True,
                        default_value="default",
                    ),
                    comp.str_attr(
                        name="d",
                        desc="",
                        is_list=False,
                        is_optional=True,
                        default_value="default",
                    ),
                ],
            ),
            comp.str_attr(
                name="e",
                desc="",
                is_list=False,
                is_optional=True,
                default_value="default",
            ),
        ],
    )

    # error
    with pytest.raises(
        EvalParamError,
        match="union group a: one attr is required, but god 'b' and 'e'.",
    ):
        EvalParamReader(instance_error, comp.definition())

    # ok
    reader = EvalParamReader(instance_ok, comp.definition())
    assert (
        reader.get_attr("a/b/c") == "c"
        and reader.get_attr("a/b/d") == "d"
        and reader.get_attr("a/e") is None
    )


def test_node_reader_param_not_set():
    instance = NodeEvalParam(
        domain="domain_a",
        name="x",
        version="v1",
    )
    definition = ComponentDef(
        domain="domain_a",
        name="x",
        version="v1",
        attrs=[
            AttributeDef(
                name="a",
                type=AttrType.AT_INT,
                atomic=AttributeDef.AtomicAttrDesc(is_optional=False),
            )
        ],
    )
    with pytest.raises(EvalParamError, match="is not optional and not set."):
        EvalParamReader(instance, definition)


def test_node_reader_get_param():
    instance = NodeEvalParam(
        domain="domain_a",
        name="x",
        version="v1",
        attrs=[
            Attribute(s="str"),
            Attribute(i64s=[13, 24]),
            Attribute(is_na=True),
        ],
        attr_paths=["a", "b", "d"],
    )
    definition = ComponentDef(
        domain="domain_a",
        name="x",
        version="v1",
        attrs=[
            AttributeDef(
                name="a",
                type=AttrType.AT_STRING,
            ),
            AttributeDef(
                name="b",
                type=AttrType.AT_INTS,
            ),
            AttributeDef(
                name="c",
                type=AttrType.AT_FLOAT,
                atomic=AttributeDef.AtomicAttrDesc(
                    is_optional=True,
                    default_value=Attribute(f=1.5),
                ),
            ),
            AttributeDef(
                name="d",
                type=AttrType.AT_FLOAT,
                atomic=AttributeDef.AtomicAttrDesc(
                    is_optional=True,
                    default_value=Attribute(f=3.0),
                ),
            ),
        ],
    )

    reader = EvalParamReader(instance, definition)

    assert reader.get_attr("a") == "str"
    assert reader.get_attr("b") == [13, 24]
    assert reader.get_attr("c") == 1.5
    assert reader.get_attr("d") == 3.0


def test_node_reader_get_io():
    instance = NodeEvalParam(
        domain="domain_a",
        name="x",
        version="v1",
        inputs=[
            DistData(name="a", type="table"),
            DistData(name="b", type="rule"),
        ],
        output_uris=["path/to/c.txt", "path/to/d.txt"],
        attr_paths=["input/a/feature"],
        attrs=[Attribute(ss=["a", "b", "c"])],
    )

    definition = ComponentDef(
        domain="domain_a",
        name="x",
        version="v1",
        inputs=[
            IoDef(
                name="a",
                types=["table"],
                attrs=[IoDef.TableAttrDef(name="feature", col_min_cnt_inclusive=1)],
            ),
            IoDef(name="b", types=["rule"]),
        ],
        outputs=[
            IoDef(name="c", types=["model"]),
            IoDef(name="d", types=["rule"]),
        ],
    )

    reader = EvalParamReader(instance, definition)
    assert reader.get_input("a") == DistData(name="a", type="table")
    assert reader.get_input("b") == DistData(name="b", type="rule")
    assert reader.get_output_uri("c") == "path/to/c.txt"
    assert reader.get_output_uri("d") == "path/to/d.txt"
    assert reader.get_input_attrs("a", "feature") == ["a", "b", "c"]
