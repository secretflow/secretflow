from secretflow.component.entry import gen_key
from secretflow.component.i18n import ROOT, gettext
from secretflow.spec.v1.component_pb2 import (
    AttributeDef,
    CompListDef,
    ComponentDef,
    IoDef,
)


def test_gettext():
    comp_list = CompListDef(
        name="secretflow",
        desc="test",
        version="0.0.1",
        comps=[
            ComponentDef(
                domain="domain",
                name="comp",
                desc="comp desc",
                version="comp_version",
                attrs=[AttributeDef(name="attr", desc="attr desc")],
                inputs=[
                    IoDef(
                        name="input",
                        desc="input desc",
                        attrs=[
                            IoDef.TableAttrDef(
                                name="key",
                                desc="key desc",
                                extra_attrs=[
                                    AttributeDef(name="key attr", desc="key attr desc")
                                ],
                            )
                        ],
                    )
                ],
                outputs=[
                    IoDef(
                        name="output",
                        desc="output desc",
                    )
                ],
            )
        ],
    )

    archives = {
        ROOT: {"secretflow": "隐语", "dummy": "dummy"},
        gen_key("domain", "comp", "comp_version"): {
            "comp": "组件",
            "input": "输入",
            "output": "输出",
            "key desc": "主键描述",
        },
    }

    text = gettext(comp_list, archives)

    assert text == {
        ROOT: {"secretflow": "隐语", "test": ""},
        gen_key("domain", "comp", "comp_version"): {
            "domain": "",
            "comp": "组件",
            "comp desc": "",
            "comp_version": "",
            "attr": "",
            "attr desc": "",
            "input": "输入",
            "input desc": "",
            "key": "",
            "key desc": "主键描述",
            "key attr": "",
            "key attr desc": "",
            "output": "输出",
            "output desc": "",
        },
    }
