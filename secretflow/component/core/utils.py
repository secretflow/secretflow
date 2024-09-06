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

import uuid
from typing import Any, Union

import secretflow.compute as sc
from secretflow.device import PYU, reveal
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from .common.utils import to_attribute
from .dist_data.vtable import VTable


def uuid4(pyu: PYU | str):
    if isinstance(pyu, str):
        pyu = PYU(pyu)
    return reveal(pyu(lambda: str(uuid.uuid4()))())


def float_almost_equal(
    a: Union[sc.Array, float], b: Union[sc.Array, float], epsilon: float = 1e-07
) -> sc.Array:
    return sc.less(sc.abs(sc.subtract(a, b)), epsilon)


def build_node_eval_param(
    domain: str,
    name: str,
    version: str,
    attrs: dict[str, Any],
    inputs: list[DistData | VTable],
    output_uris: list[str],
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
    )
    return param
