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

import math
import re
import uuid
from typing import Any

import secretflow.compute as sc
from secretflow.device import PYU, reveal
from secretflow.spec.v1.component_pb2 import CompListDef
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from .common.utils import to_attribute
from .dist_data.vtable import VTable


def uuid4(pyu: PYU | str):
    if isinstance(pyu, str):
        pyu = PYU(pyu)
    return reveal(pyu(lambda: str(uuid.uuid4()))())


def float_almost_equal(
    a: sc.Array | float, b: sc.Array | float, epsilon: float = 1e-07
) -> sc.Array:
    return sc.less(sc.abs(sc.subtract(a, b)), epsilon)


def pad_inf_to_split_points(split_points: list[float]) -> list[float]:
    assert isinstance(split_points, list), f"{split_points}"
    return [-math.inf] + split_points + [math.inf]


def build_node_eval_param(
    domain: str,
    name: str,
    version: str,
    attrs: dict[str, Any],
    inputs: list[DistData | VTable],
    output_uris: list[str],
    checkpoint_uri: str | None = None,
) -> NodeEvalParam:  # type: ignore
    '''
    Used for constructing NodeEvalParam in unit tests.
    '''

    if attrs:
        attr_paths = []
        attr_values = []
        for k, v in attrs.items():
            attr_paths.append(k)
            attr_values.append(to_attribute(v))
    else:
        attr_paths = None
        attr_values = None

    dd_inputs = None
    if inputs:
        dd_inputs = []
        for item in inputs:
            if isinstance(item, DistData):
                dd_inputs.append(item)
            elif isinstance(item, VTable):
                dd_inputs.append(item.to_distdata())
            else:
                raise ValueError(f"invalid DistData type, {type(item)}")

    param = NodeEvalParam(
        domain=domain,
        name=name,
        version=version,
        attr_paths=attr_paths,
        attrs=attr_values,
        inputs=dd_inputs,
        output_uris=output_uris,
        checkpoint_uri=checkpoint_uri,
    )
    return param


def gen_key(domain: str, name: str, version: str) -> str:
    return f"{domain}/{name}:{version}"


# The format of the gettext output:
# {
#     'comp': {
#         'text': 'archieved translation or empty'
#         '...':'...'
#     },
#     '...':{}
# }
def gettext(comp_list: CompListDef, archives=None):  # type: ignore
    def restore_from_archives(text, key, archives=None):
        if archives is None:
            return text

        if key not in archives:
            return text
        archive = archives[key]
        for k in text.keys():
            if k in archive:
                text[k] = archive[k]

        return text

    ROOT = "."

    ret = {}
    root_text = {}
    root_text[comp_list.name] = ""
    root_text[comp_list.desc] = ""

    ret[ROOT] = restore_from_archives(root_text, ROOT, archives)

    for comp in comp_list.comps:
        text = {}
        text[comp.domain] = ""
        text[comp.name] = ""
        text[comp.desc] = ""
        text[comp.version] = ""

        for attr in comp.attrs:
            text[attr.name] = ""
            text[attr.desc] = ""

        for io in list(comp.inputs) + list(comp.outputs):
            text[io.name] = ""
            text[io.desc] = ""

            for t_attr in io.attrs:
                text[t_attr.name] = ""
                text[t_attr.desc] = ""
                for t_attr_a in t_attr.extra_attrs:
                    text[t_attr_a.name] = ""
                    text[t_attr_a.desc] = ""

        key = gen_key(comp.domain, comp.name, comp.version)
        ret[key] = restore_from_archives(text, key, archives)

    return ret


LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+")
TWO_LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+((\r\n)|[\n\v])+")
MULTI_WHITESPACE_TO_ONE_REGEX = re.compile(r"\s+")
NONBREAKING_SPACE_REGEX = re.compile(r"(?!\n)\s+")


def normalize_whitespace(
    text: str, no_line_breaks=False, strip_lines=True, keep_two_line_breaks=False
):
    """
    Given ``text`` str, replace one or more spacings with a single space, and one
    or more line breaks with a single newline. Also strip leading/trailing whitespace.
    """
    if strip_lines:
        text = "\n".join([x.strip() for x in text.splitlines()])

    if no_line_breaks:
        text = MULTI_WHITESPACE_TO_ONE_REGEX.sub(" ", text)
    else:
        if keep_two_line_breaks:
            text = NONBREAKING_SPACE_REGEX.sub(
                " ", TWO_LINEBREAK_REGEX.sub(r"\n\n", text)
            )
        else:
            text = NONBREAKING_SPACE_REGEX.sub(" ", LINEBREAK_REGEX.sub(r"\n", text))

    return text.strip()


DOUBLE_QUOTE_REGEX = re.compile("|".join("«»“”„‟‹›❝❞❮❯〝〞〟＂"))
SINGLE_QUOTE_REGEX = re.compile("|".join("`´‘‘’’‛❛❜"))


def fix_strange_quotes(text):
    """
    Replace strange quotes, i.e., 〞with a single quote ' or a double quote " if it fits better.
    """
    text = SINGLE_QUOTE_REGEX.sub("'", text)
    text = DOUBLE_QUOTE_REGEX.sub('"', text)
    return text


def clean_text(text: str, no_line_breaks: bool = True) -> str:
    text = text.strip()
    text = normalize_whitespace(text, no_line_breaks)
    text = fix_strange_quotes(text)
    return text
