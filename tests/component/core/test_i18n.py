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


from secretflow_spec.v1.component_pb2 import (
    AttributeDef,
    CompListDef,
    ComponentDef,
    IoDef,
)

from secretflow.component.core import get_translation, translate
from secretflow.component.core.i18n import gettext


def test_translate():
    res = translate("secretflow", None)
    assert len(res) > 0 and "." in res

    res = get_translation()
    assert len(res) > 0 and "." in res


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
                version="0.0.1",
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

    ROOT = "."
    COMP_ID = "domain/comp:0.0.1"

    archives = {
        ROOT: {"secretflow": "隐语", "dummy": "dummy"},
        COMP_ID: {
            "comp": "组件",
            "input": "输入",
            "output": "输出",
            "key desc": "主键描述",
        },
    }

    text = gettext(comp_list, archives, build_root=True)

    assert text == {
        ROOT: {"secretflow": "隐语", "test": ""},
        COMP_ID: {
            "domain": "",
            "comp": "组件",
            "comp desc": "",
            "0.0.1": "",
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
