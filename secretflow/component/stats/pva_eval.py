# Copyright 2023 Ant Group Co., Ltd.
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

from secretflow.component.component import Component, IoType
from secretflow.component.data_utils import (
    DistDataType,
    dump_pva_eval_result,
    load_table,
)
from secretflow.device.driver import reveal
from secretflow.stats.pva_eval import pva_eval

pva_value_comp = Component(
    "pva_eval",
    domain="stats",
    version="0.0.1",
    desc="Compute Prediction Vs Actual score, i.e. abs(mean(prediction) - sum(actual == target)/count(actual))",
)
pva_value_comp.float_attr(
    name="target",
    desc="The target value.",
    is_list=False,
    is_optional=False,
)
pva_value_comp.io(
    io_type=IoType.INPUT,
    name="actual",
    desc="Actual score.",
    types=[DistDataType.VERTICAL_TABLE],
)
pva_value_comp.io(
    io_type=IoType.INPUT,
    name="prediction",
    desc="Prediction score.",
    types=[DistDataType.VERTICAL_TABLE],
)
pva_value_comp.io(
    io_type=IoType.OUTPUT,
    name="result",
    desc="Output report.",
    types=[DistDataType.REPORT],
)


@pva_value_comp.eval_fn
def pva_eval_fn(
    *,
    ctx,
    target,
    actual,
    prediction,
    result,
):
    actual_data = load_table(
        ctx, actual, load_features=True, load_ids=True, load_labels=True
    )
    prediction_data = load_table(
        ctx, prediction, load_features=True, load_ids=True, load_labels=True
    )

    with ctx.tracer.trace_running():
        res = reveal(
            pva_eval(actual=actual_data, prediction=prediction_data, target=target)
        )

    return {"result": dump_pva_eval_result(result, actual.sys_info, res)}
