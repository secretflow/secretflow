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

import numpy as np

from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import DistDataType, load_table
from secretflow.device.device.spu import SPU
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Div, Report, Tab, Table
from secretflow.stats.ss_pearsonr_v import PearsonR

ss_pearsonr_comp = Component(
    "ss_pearsonr",
    domain="stats",
    version="0.0.1",
    desc="""Calculate Pearson's product-moment correlation coefficient for vertical partitioning dataset
    by using secret sharing.

    - For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
    """,
)
ss_pearsonr_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="feature_selects",
            desc="Specify which features to calculate correlation coefficient with. If empty, all features will be used",
        )
    ],
)
ss_pearsonr_comp.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="Output Pearson's product-moment correlation coefficient report.",
    types=[DistDataType.REPORT],
)


@ss_pearsonr_comp.eval_fn
def ss_pearsonr_eval_fn(
    *,
    ctx,
    input_data,
    input_data_feature_selects,
    report,
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    feature_selects = (
        input_data_feature_selects if len(input_data_feature_selects) else None
    )

    x = load_table(
        ctx,
        input_data,
        load_features=True,
        feature_selects=feature_selects,
    )

    with ctx.tracer.trace_running():
        pr: np.ndarray = PearsonR(spu).pearsonr(x)

    feature_names = x.columns

    assert pr.shape[0] == len(feature_names) and pr.shape[1] == len(feature_names)

    r_table = Table(
        headers=[
            Table.HeaderItem(name=f, desc="", type="float") for f in feature_names
        ],
        rows=[
            Table.Row(
                name=feature_names[r], desc="", items=[Attribute(f=c) for c in pr[r]]
            )
            for r in range(pr.shape[0])
        ],
    )

    report_mate = Report(
        name="corr",
        desc="corr table",
        tabs=[
            Tab(
                divs=[
                    Div(
                        children=[
                            Div.Child(
                                type="table",
                                table=r_table,
                            )
                        ],
                    )
                ],
            )
        ],
    )

    report_dd = DistData(
        name=report,
        type=str(DistDataType.REPORT),
        system_info=input_data.system_info,
    )
    report_dd.meta.Pack(report_mate)

    return {"report": report_dd}
